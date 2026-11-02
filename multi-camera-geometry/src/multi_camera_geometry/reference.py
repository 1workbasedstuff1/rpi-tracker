import torch
import torch.nn.functional as F
from dataclasses import dataclass
import multi_camera_geometry.helper as hp
import numpy as np
from scipy.spatial.transform import Rotation

"""
the IMU + MAG + GPS produces a state  
nominal_state: [p, v, q, a_b, ω_b]
p - postion in global cartesian (x, y, z) (3 states)
v - velocity in global frame (3 states)
q - quaternion orientation (4 states)
a_b - accelerometer bias (3 states)
ω_b - gyrometer bias (3 states)

and the error covariance matrix is
Error state: [δp, δv, δθ, δa_b δω_b]
so Error matrix is a 15x15 error matrix
"""

@dataclass 
class PoseNED:
    # holds a state: [x q] column vector that is 
    # that is a 1x7
    # global position and orientation
    # q is a local-global quaternion
    state: np.ndarray

class PoseErrorNED:
    # holds a 6x6 covariance matrix 
    # [Cpp Cpθ]
    # [Cθp Cθθ]
    covariance: np.ndarray

"""
This function should take a 3D point defined in NED coordinates
(x: North, y: East: z: Down)
and project them into our camera plane
    (A possible issue is that I think camera frame has 
        Z axis - Pointing out from image plane centre
        Y axis - Pointing down to the camera
        X axit - Pointing right of camera
    )
NOTE: our IMU FRD, where the X axis aligns with the Z axis of the 
        camera (both point forward) 

TRANSFORM: 
    - First we rotate the NED coordinates 90 degrees about the y axis
      so that x points up and z points forward
    - Then we rotate 90 degrees clockwise about the z axis so that 
      x points left and y points down

3D COORDINATE -> IMAGE PLANE:
    h(X, K, R, c) = π (K X_c)
"""
def ned2cameraframe(x_ned: np.ndarray, pose: PoseNED, intrinsic_matrix: np.ndarray):
    if x_ned.shape != (3,):
        raise ValueError("x_ned is 3d coordinate!")
    if intrinsic_matrix.shape != (3,3):
        raise ValueError("intrinsic_matrix is a 3x3 matrix for calibrating camera")

    # transform the x_ned coordinates into the camera frame
    position = pose.state[0:3]
    quaternion = pose.state[3:7].astype(float)
    quaternion = quaternion / np.linalg.norm(quaternion)

    # SECTION 3: ROTATE NED -> BODY FRD
    # A normalized quaternion inverse is its conjugate. Convert the stored
    # body -> NED quaternion into the NED -> body rotation needed here.
    quaternion_inv = quaternion.copy()
    quaternion_inv[1:] *= -1


    # perform transform: 
    # X_c = R (X_ned - c) where i think c is estimated position from kalman filter 
    # and R = R{q}? this would be local to global
    X_body_frd = hp.quaternion_rotate(quaternion_inv, x_ned - position)

    # SECTION 4: ROTATE BODY FRD -> CAMERA
    # Body axes [forward, right, down] map to camera axes
    # [right, down, forward].
    bodyfrd2camera = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    X_c = bodyfrd2camera @ X_body_frd

    image_plane_xy = hp.project_π(intrinsic_matrix @ X_c)

    return image_plane_xy


def ned2cameraframe_tensor(
    x_ned: torch.Tensor,
    pose: PoseNED,
    intrinsic_matrix: torch.Tensor
):
    """
    Project one global NED point into the camera image using PyTorch.

    Args:
        x_ned: (3,) landmark position in global North-East-Down coordinates.
        pose: Camera position and body-FRD-to-NED quaternion [w,x,y,z].
        intrinsic_matrix: (3,3) camera calibration matrix K.

    Returns:
        (2,) projected pixel coordinate.

    The complete transform is

        X_body = R_bn(q) (X_ned - position)
        X_camera = R_cb X_body
        pixel = pi(K X_camera).

    Every calculation remains in the PyTorch graph, so the result can be
    differentiated with respect to x_ned and tensor-valued pose inputs.
    """
    if x_ned.shape != (3,):
        raise ValueError("x_ned is 3d coordinate!")
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError("intrinsic_matrix is a 3x3 matrix for calibrating camera")
    if not isinstance(x_ned, torch.Tensor) or not torch.is_floating_point(x_ned):
        raise TypeError("x_ned must be a floating-point torch.Tensor")

    # SECTION 1: STANDARDIZE THE INPUTS
    # Keep every input on the same dtype and device as x_ned.
    state = torch.as_tensor(pose.state, dtype=x_ned.dtype, device=x_ned.device)
    intrinsic_matrix = torch.as_tensor(
        intrinsic_matrix, dtype=x_ned.dtype, device=x_ned.device
    )

    # SECTION 2: TRANSLATE GLOBAL NED -> CAMERA-CENTRED NED
    # Subtracting the optical centre expresses the landmark relative to this
    # camera. q describes the body FRD -> global NED orientation.
    position = state[0:3]
    quaternion = F.normalize(state[3:7], dim=0)

    # SECTION 3: ROTATE NED -> BODY FRD
    # A normalized quaternion inverse is its conjugate. Convert the stored
    # body -> NED quaternion into the NED -> body rotation needed here.
    quaternion_inv = hp.quaternion_conjugate_tensor(quaternion)

    # Apply X_body = R_bn(q) (X_ned - position).
    X_body_frd = hp.quaternion_rotate_tensor(
        quaternion_inv, x_ned - position
    )

    # SECTION 4: ROTATE BODY FRD -> CAMERA
    # Body axes [forward, right, down] map to camera axes
    # [right, down, forward].
    bodyfrd2camera = x_ned.new_tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    X_c = bodyfrd2camera @ X_body_frd

    # SECTION 5: APPLY INTRINSICS AND PERSPECTIVE DIVISION
    # K maps the camera point to homogeneous image coordinates. pi then
    # divides by the third coordinate to produce the final pixel [u, v].
    image_plane_xy = hp.project_π_tensor(intrinsic_matrix @ X_c)

    return image_plane_xy


"""
We're given multiple image coordinates and need to provide 
an initial estimate to begin optimising the estimation of the 
3d coordinate

residual error
ri = zi - h(X, Ki, Ri, ci)

zi - detecting point coordinates from camera i
X - estimated 3d position
Ki - intrinsic matrix for camera i
Ri - orientation of camera i given by R{q_{localfrd -> globalned}}
ci - position of camera in NED frame
h - ned2cameraframe(x_ned, [ci Ri], Ki)

our objective is to find X which minimises the residual
X = min Σ||ri(X)||^2

eventually we want to minimise 
X* = arg min X Σ ri(X)T Si^-1(X) ri(X) + log(det(Si(X)))
but this will require an initial estimate for X
"""

def initial_3d_estimate_tensor(image_points, states, Ks):
    """
    Estimate the initial 3D position of a landmark in NED coordinates
    using linear triangulation from multiple camera observations.

    Each camera provides:
        1. Its position p_i in NED.
        2. Its orientation q_i (body FRD -> NED).
        3. An observed image coordinate z_i = [u_i, v_i].
        4. Its intrinsic camera matrix K_i.

    The image observation defines a 3D ray:

        L_i(lambda) = p_i + lambda * d_i

    where:
        p_i = camera position in NED (3,)
        d_i = unit direction vector in NED (3,)
        lambda >= 0 = distance along the ray

    With multiple cameras, these rays should intersect at the
    landmark X. However, measurement noise and pose uncertainty
    mean that the rays generally do not intersect exactly.

    We therefore find the point X that minimises the sum of
    squared perpendicular distances to the supporting lines:

        X* = argmin_X sum_i ||P_i @ (X - p_i)||^2

    where:
        P_i = I - d_i @ d_i.T

    The resulting least-squares solution satisfies:

        A @ X = b

    with:
        A = sum_i P_i
        b = sum_i P_i @ p_i

    Inputs:
        image_points: (N, 2) observed pixel coordinates
        states:       (N, 7) camera poses [p, q] in NED
        Ks:           (N, 3, 3) camera intrinsic matrices

    Returns:
        (3,) initial landmark estimate in NED
    """

    # Use image_points as the source of the common dtype and device.
    # For BFGS, pass float64 tensors for the best numerical accuracy.
    if isinstance(image_points, torch.Tensor) and torch.is_floating_point(image_points):
        dtype = image_points.dtype
        device = image_points.device
    else:
        dtype = torch.float64
        device = None

    image_points = torch.as_tensor(image_points, dtype=dtype, device=device)
    states = torch.as_tensor(states, dtype=dtype, device=device)
    Ks = torch.as_tensor(Ks, dtype=dtype, device=device)

    # -------------------------------------------------------------
    # CAMERA TO BODY COORDINATE TRANSFORMATION
    # -------------------------------------------------------------
    #
    # Camera frame:
    #   x = right
    #   y = down
    #   z = forward (optical axis)
    #
    # Body FRD frame:
    #   x = forward
    #   y = right
    #   z = down
    #
    # Therefore, a vector in camera coordinates:
    #
    #       d_camera = [x_c, y_c, z_c]
    #
    # becomes, in body FRD coordinates:
    #
    #       d_body = [z_c, x_c, y_c]
    #
    # This transformation is:
    #
    #       d_body = R_cb @ d_camera
    #
    # where R_cb rotates camera-frame vectors into body FRD.
    #
    R_cb = image_points.new_tensor([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
    ])

    # -------------------------------------------------------------
    # INITIALISE THE LINEAR LEAST-SQUARES SYSTEM
    # -------------------------------------------------------------
    #
    # We want to solve:
    #
    #       A @ X = b
    #
    # where X is the unknown landmark position in NED.
    #
    # Each camera contributes:
    #
    #       A_i = I - d_i @ d_i.T
    #       b_i = A_i @ p_i
    #
    # Summing over all cameras:
    #
    #       A = sum_i A_i
    #       b = sum_i b_i
    #
    # Note: A is a 3x3 triangulation geometry matrix,
    # not a camera intrinsic or projection matrix.
    #
    A = image_points.new_zeros((3, 3))
    b = image_points.new_zeros(3)

    I = torch.eye(3, dtype=dtype, device=image_points.device)

    for uv, state, K in zip(image_points, states, Ks):

        # Camera position in NED, assumed to be the optical centre.
        p = state[:3]

        # Quaternion rotating body FRD -> global NED.
        q = F.normalize(state[3:7], dim=0)

        # ---------------------------------------------------------
        # STEP 1: IMAGE PIXEL -> 3D DIRECTION IN CAMERA FRAME
        # ---------------------------------------------------------
        #
        # Our observed image coordinate is:
        #
        #       z_i = [u_i, v_i]
        #
        # Convert it to homogeneous coordinates:
        #
        #       z_h = [u_i, v_i, 1]^T
        #
        # The pinhole camera model is:
        #
        #       z_h ~ K_i @ X_camera
        #
        # where ~ means equality up to a scale factor.
        #
        # Rearranging gives a direction in camera coordinates:
        #
        #       d_camera = K_i^(-1) @ z_h
        #
        # This is a 3D direction, NOT a 3D landmark position.
        #
        # For a standard intrinsic matrix:
        #
        #       K = [fx  s  cx]
        #           [ 0 fy  cy]
        #           [ 0  0   1]
        #
        # and zero skew (s = 0), this corresponds to:
        #
        #       d_camera = [(u-cx)/fx, (v-cy)/fy, 1]
        #
        # The direction points from the camera optical centre
        # towards the observed image feature.
        #
        pixel_h = torch.cat([uv, uv.new_ones(1)])

        # Equivalent to inv(K) @ pixel_h, but more stable.
        ray_camera = torch.linalg.solve(K, pixel_h)

        # ---------------------------------------------------------
        # STEP 2: CAMERA DIRECTION -> BODY FRD DIRECTION
        # ---------------------------------------------------------
        #
        # The direction is currently expressed in camera axes.
        #
        # Apply our fixed camera-to-body rotation:
        #
        #       d_body = R_cb @ d_camera
        #
        ray_body = R_cb @ ray_camera

        # ---------------------------------------------------------
        # STEP 3: BODY FRD DIRECTION -> GLOBAL NED DIRECTION
        # ---------------------------------------------------------
        #
        # The camera orientation is represented by quaternion q,
        # which rotates vectors from body FRD to global NED.
        #
        # In rotation matrix notation:
        #
        #       d_ned = R_nb(q) @ d_body
        #
        # or, combining all transformations:
        #
        #       d_ned = R_nb(q) @ R_cb @ K_i^(-1) @ z_h
        #
        # This gives the ray direction in the global NED frame.
        #
        # Normalize so that:
        #
        #       ||d_ned|| = 1
        #
        # The final direction is:
        #
        #       d_i = d_ned / ||d_ned||
        #
        # The 3D line defined by camera i is now:
        #
        #       L_i(lambda) = p_i + lambda * d_i
        #
        ray_ned = hp.quaternion_rotate_tensor(q, ray_body)
        ray_ned = F.normalize(ray_ned, dim=0)

        # ---------------------------------------------------------
        # STEP 4: CONSTRUCT THE PERPENDICULAR PROJECTION MATRIX
        # ---------------------------------------------------------
        #
        # We now have a camera position p_i and unit direction d_i.
        #
        # For an arbitrary landmark estimate X, define:
        #
        #       v_i = X - p_i
        #
        # This vector can be separated into two components:
        #
        #   1. Parallel to the camera ray:
        #
        #       v_parallel = (d_i @ d_i.T) @ v_i
        #
        #   2. Perpendicular to the camera ray:
        #
        #       v_perp = v_i - v_parallel
        #
        # Therefore:
        #
        #       v_perp = (I - d_i @ d_i.T) @ (X - p_i)
        #
        # Define the perpendicular projection matrix:
        #
        #       P_i = I - d_i @ d_i.T
        #
        # Since d_i is a unit vector, P_i is symmetric and
        # idempotent (P_i.T = P_i, P_i @ P_i = P_i).
        #
        # It removes the component parallel to the ray,
        # leaving only the perpendicular error.
        #
        # Consequently:
        #
        #       ||P_i @ (X - p_i)||^2
        #
        # is the squared perpendicular distance from X to
        # the infinite line passing through camera i.
        #
        P = I - torch.outer(ray_ned, ray_ned)

        # ---------------------------------------------------------
        # STEP 5: ACCUMULATE THE LEAST-SQUARES SYSTEM
        # ---------------------------------------------------------
        #
        # We wish to find X that minimises:
        #
        #       E(X) = sum_i ||P_i @ (X - p_i)||^2
        #
        # Expanding each squared norm:
        #
        #       E(X) = sum_i (X - p_i).T @ P_i @ (X - p_i)
        #
        # This simplification is valid because P_i is
        # symmetric and idempotent.
        #
        # Differentiate with respect to X:
        #
        #       dE/dX = 2 * sum_i P_i @ (X - p_i)
        #
        # Set the derivative to zero:
        #
        #       sum_i P_i @ (X - p_i) = 0
        #
        # Rearrange:
        #
        #       (sum_i P_i) @ X = sum_i P_i @ p_i
        #
        # This is a linear system:
        #
        #       A @ X = b
        #
        # where:
        #
        #       A = sum_i P_i
        #       b = sum_i P_i @ p_i
        #
        # Each camera therefore contributes constraints on X
        # in the two directions perpendicular to its ray.
        #
        # With sufficient parallax between camera rays,
        # the combined system constrains all three coordinates.
        #
        A += P
        b += P @ p

    # -------------------------------------------------------------
    # STEP 6: CHECK WHETHER TRIANGULATION IS WELL CONSTRAINED
    # -------------------------------------------------------------
    #
    # If all camera rays are parallel or nearly parallel,
    # the 3D landmark position is poorly constrained.
    #
    # In particular, the depth along the shared viewing
    # direction cannot be accurately determined.
    #
    # In this case, A becomes singular or ill-conditioned.
    #
    # The condition number:
    #
    #       cond(A) = largest_singular_value(A)
    #                 / smallest_singular_value(A)
    #
    # indicates how sensitive the solution is to noise.
    #
    # A very large condition number means that small changes
    # in observations can produce large changes in X.
    #
    if torch.linalg.cond(A) > 1e8:
        raise ValueError(
            "Degenerate triangulation: insufficient parallax"
        )

    # -------------------------------------------------------------
    # STEP 7: SOLVE FOR THE INITIAL LANDMARK POSITION
    # -------------------------------------------------------------
    #
    # Solve the normal equations:
    #
    #       A @ X = b
    #
    # Therefore:
    #
    #       X = A^(-1) @ b
    #
    # Rather than explicitly computing A^(-1), use
    # torch.linalg.solve for better numerical stability.
    #
    # The result is the landmark position in global NED.
    #
    return torch.linalg.solve(A, b)


def initial_3d_estimate(image_points, states, Ks):
    """Call the tensor triangulation implementation using the original name."""
    return initial_3d_estimate_tensor(image_points, states, Ks)


"""
We need to compute the matrix Si which is the total 2d image position 
uncertainty. It accounts for both uncertainty in the image detection and 
tracking and also the uncertainty causes by position and velocity

Si = Σpixel + Jpose Σpose Jpose 

Jpose = dh/d(δp,δθ)
where h is ned2cameraframe(Xned, qi, pi, Ki)

h = π(Ki R_cb R^-1(qi) (Xned - pi))

and differentiating gives 

h = π(Ki Xc) 

dh/dXc * dXc/d(δp,δθ)

giving 

Jπ * [-R_cb * R_bn   R_cb [X_b]x]

where 
Xb = R_bn(Xned - pi)

"""
def ned2pixel_with_covariance(
    x_ned: np.ndarray,
    pose: PoseNED,
    intrinsic_matrix: np.ndarray,
    Sigma_pose: np.ndarray,
    Sigma_pixel: np.ndarray
):
    """
    Project a NED point into the camera and propagate pose uncertainty.

    Sigma_pose: (6,6) covariance ordered as:
        [dp_N, dp_E, dp_D, dtheta_bx, dtheta_by, dtheta_bz]

    Position errors are in NED.
    Orientation errors are right-multiplicative, in body FRD.
    Angular errors must be in radians.

    Returns:
        uv: (2,) projected pixel coordinates
        S: (2,2) total pixel covariance
        J_pose: (2,6) pose-to-pixel Jacobian
    """

    p = np.asarray(pose.state[:3], dtype=float)

    # Quaternion [w, x, y, z], body -> NED
    q = np.asarray(pose.state[3:7], dtype=float)
    q /= np.linalg.norm(q)

    # scipy expects [x, y, z, w]
    R_nb = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    R_bn = R_nb.T

    R_cb = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=float)

    # 1. NED -> Body -> Camera
    X_b = R_bn @ (x_ned - p)
    X_c = R_cb @ X_b

    if X_c[2] <= 0:
        raise ValueError("Point is behind the camera")

    # 2. Project into pixel coordinates
    y = intrinsic_matrix @ X_c
    a, b, w = y

    uv = np.array([a / w, b / w])

    # 3. Projection Jacobian
    # Supports a general 3x3 intrinsic matrix.
    J_pi = np.array([
        [1 / w, 0, -a / (w * w)],
        [0, 1 / w, -b / (w * w)]
    ]) @ intrinsic_matrix

    # 4. Pose Jacobian
    J_position = -R_cb @ R_bn
    J_orientation = R_cb @ hp.skew(X_b)

    J_pose = J_pi @ np.hstack([
        J_position,
        J_orientation
    ])

    # 5. Propagate pose covariance into pixel space
    S = Sigma_pixel + J_pose @ Sigma_pose @ J_pose.T

    # Remove small numerical asymmetry
    S = 0.5 * (S + S.T)

    return uv, S, J_pose


def ned2pixel_with_covariance_tensor(
    x_ned: torch.Tensor,
    pose: PoseNED,
    intrinsic_matrix: torch.Tensor,
    Sigma_pose: torch.Tensor,
    Sigma_pixel: torch.Tensor
):
    """
    Project a NED point into the camera and propagate pose uncertainty.

    This is the PyTorch/autograd equivalent of
    ned2pixel_with_covariance().

    Sigma_pose: (6,6) covariance ordered as:
        [dp_N, dp_E, dp_D, dtheta_bx, dtheta_by, dtheta_bz]

    Position errors are in NED.
    Orientation errors are right-multiplicative, in body FRD.
    Angular errors must be in radians.

    Returns:
        uv: (2,) projected pixel coordinates
        S: (2,2) total pixel covariance
        J_pose: (2,6) pose-to-pixel Jacobian

    Since X_b, X_c, J_pi, J_pose and S are all built using PyTorch
    operations, autograd includes both the changing residual and the
    changing covariance when differentiating the likelihood objective.
    """
    if not isinstance(x_ned, torch.Tensor) or not torch.is_floating_point(x_ned):
        raise TypeError("x_ned must be a floating-point torch.Tensor")
    if x_ned.shape != (3,):
        raise ValueError("x_ned must have shape (3,)")

    # Keep every input on the same dtype and device as x_ned. This avoids
    # accidental CPU/float32 constants breaking a CUDA/float64 calculation.
    state = torch.as_tensor(pose.state, dtype=x_ned.dtype, device=x_ned.device)
    intrinsic_matrix = torch.as_tensor(
        intrinsic_matrix, dtype=x_ned.dtype, device=x_ned.device
    )
    Sigma_pose = torch.as_tensor(
        Sigma_pose, dtype=x_ned.dtype, device=x_ned.device
    )
    Sigma_pixel = torch.as_tensor(
        Sigma_pixel, dtype=x_ned.dtype, device=x_ned.device
    )

    if state.shape != (7,):
        raise ValueError("pose.state must have shape (7,)")
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape (3, 3)")
    if Sigma_pose.shape != (6, 6):
        raise ValueError("Sigma_pose must have shape (6, 6)")
    if Sigma_pixel.shape != (2, 2):
        raise ValueError("Sigma_pixel must have shape (2, 2)")

    p = state[:3]

    # Quaternion [w, x, y, z], body -> NED
    q = F.normalize(state[3:7], dim=0)

    # Construct body -> NED directly in PyTorch, then transpose it to obtain
    # NED -> body. This replaces scipy.spatial.transform.Rotation.
    R_nb = hp.quaternion_to_rotation_matrix_tensor(q)
    R_bn = R_nb.transpose(-1, -2)

    R_cb = x_ned.new_tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    # 1. NED -> Body -> Camera
    X_b = R_bn @ (x_ned - p)
    X_c = R_cb @ X_b

    # This check only controls whether the projection is geometrically valid.
    # It does not modify or detach X_c in the calculations below.
    if X_c[2].detach().item() <= 0:
        raise ValueError("Point is behind the camera")

    # 2. Project into pixel coordinates
    y = intrinsic_matrix @ X_c
    a, b, w = y.unbind()

    uv = torch.stack([a / w, b / w])

    # 3. Projection Jacobian
    # Supports a general 3x3 intrinsic matrix. torch.stack is important here:
    # torch.tensor([a, b]) would copy and detach a and b from autograd.
    zero = torch.zeros_like(w)
    J_pi = torch.stack([
        torch.stack([1.0 / w, zero, -a / (w * w)]),
        torch.stack([zero, 1.0 / w, -b / (w * w)])
    ]) @ intrinsic_matrix

    # 4. Pose Jacobian
    J_position = -R_cb @ R_bn
    J_orientation = R_cb @ hp.skew_tensor(X_b)

    J_pose = J_pi @ torch.cat([
        J_position,
        J_orientation
    ], dim=1)

    # 5. Propagate pose covariance into pixel space
    S = Sigma_pixel + J_pose @ Sigma_pose @ J_pose.transpose(-1, -2)

    # Remove small numerical asymmetry. Unlike an in-place assignment, this
    # expression remains safe for autograd.
    S = 0.5 * (S + S.transpose(-1, -2))

    return uv, S, J_pose


"""
Evaluate the probabilistic cost of one candidate 3D point.

For camera i, the measured pixel z_i is compared with the projected pixel
h_i(X). Their residual is

    r_i(X) = z_i - h_i(X).

The 2D covariance S_i(X) combines pixel measurement uncertainty with camera
pose uncertainty propagated through the projection Jacobian. The objective is

    F(X) = sum_i [r_i.T S_i^-1 r_i + log(det(S_i))].

The first term penalizes reprojection error according to its uncertainty. The
log-determinant term prevents an uncertain projection from obtaining a low cost
simply because S_i is large. This function only evaluates F(X); the BFGS
algorithm that minimizes it is documented above estimate_3d_point_tensor().
"""
def maximum_likelihood_objective_tensor(
    X, image_points, states, Ks, Sigma_poses, Sigma_pixels
):
    """
    Evaluate the multi-camera negative log-likelihood at one candidate X.

    Args:
        X: (3,) candidate landmark position in global NED coordinates.
        image_points: (N,2) measured pixels z_i.
        states: (N,7) camera positions and body-to-NED quaternions.
        Ks: (N,3,3) camera intrinsic matrices.
        Sigma_poses: (N,6,6) pose covariances.
        Sigma_pixels: (N,2,2) pixel measurement covariances.

    Returns:
        Scalar tensor F(X), still attached to the autograd graph.

    Both the projected residual r_i(X) and covariance S_i(X) depend on X,
    and both dependencies are retained when autograd calculates dF/dX.
    """
    # SECTION 1: VALIDATE THE CANDIDATE LANDMARK
    # X must remain a floating-point tensor because it is the variable with
    # respect to which autograd differentiates the objective.
    if not isinstance(X, torch.Tensor) or not torch.is_floating_point(X):
        raise TypeError("X must be a floating-point torch.Tensor")
    if X.shape != (3,):
        raise ValueError("X must have shape (3,)")

    # SECTION 2: STANDARDIZE OBSERVATION INPUTS
    # Matching X's dtype and device prevents implicit casts or CPU/GPU
    # mismatches inside the per-camera projection loop.
    image_points = torch.as_tensor(
        image_points, dtype=X.dtype, device=X.device
    )
    states = torch.as_tensor(states, dtype=X.dtype, device=X.device)
    Ks = torch.as_tensor(Ks, dtype=X.dtype, device=X.device)
    Sigma_poses = torch.as_tensor(
        Sigma_poses, dtype=X.dtype, device=X.device
    )
    Sigma_pixels = torch.as_tensor(
        Sigma_pixels, dtype=X.dtype, device=X.device
    )

    # SECTION 3: VALIDATE THE MULTI-CAMERA BATCH SHAPES
    # All inputs share the same leading camera dimension N.
    number_of_cameras = image_points.shape[0]
    expected_shapes = {
        "image_points": (number_of_cameras, 2),
        "states": (number_of_cameras, 7),
        "Ks": (number_of_cameras, 3, 3),
        "Sigma_poses": (number_of_cameras, 6, 6),
        "Sigma_pixels": (number_of_cameras, 2, 2),
    }
    actual_values = {
        "image_points": image_points,
        "states": states,
        "Ks": Ks,
        "Sigma_poses": Sigma_poses,
        "Sigma_pixels": Sigma_pixels,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_values[name].shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {tuple(actual_values[name].shape)}"
            )

    if number_of_cameras < 2:
        raise ValueError("At least two camera observations are required")

    # SECTION 4: ACCUMULATE EACH CAMERA'S LIKELIHOOD CONTRIBUTION
    # Start with a scalar on the same dtype/device as X.
    total = X.new_zeros(())

    for z, state, K, Sigma_pose, Sigma_pixel in zip(
        image_points, states, Ks, Sigma_poses, Sigma_pixels
    ):
        # Project X into this image and propagate pose uncertainty to obtain
        # the X-dependent 2D covariance S_i(X).
        uv, S, _ = ned2pixel_with_covariance_tensor(
            X, PoseNED(state), K, Sigma_pose, Sigma_pixel
        )

        # The residual points from the predicted pixel to the measurement.
        residual = z - uv

        # solve(S, residual) is more stable than explicitly forming S^-1.
        weighted_residual = torch.linalg.solve(S, residual)
        mahalanobis_distance = residual @ weighted_residual

        # slogdet is numerically safer than log(det(S)). A valid covariance
        # must be positive definite, and therefore have a positive determinant.
        sign, log_absolute_determinant = torch.linalg.slogdet(S)
        if sign.detach().item() <= 0:
            raise ValueError("Projected pixel covariance is not positive definite")

        # Add F_i to the scalar total without detaching either term.
        total = total + mahalanobis_distance + log_absolute_determinant

    return total


"""
Refine the triangulated landmark by minimizing the maximum-likelihood cost.

The optimizer uses full BFGS because X has only three components. Starting
from X0 and H0 = I, each iteration performs these steps:

    1. Evaluate F(Xk) and its autograd gradient gk = dF/dX.
    2. Form the descent direction pk = -Hk gk.
    3. Use an Armijo backtracking line search to choose alpha_k.
    4. Accept X{k+1} = Xk + alpha_k pk.
    5. Define sk = X{k+1} - Xk and yk = g{k+1} - gk.
    6. Update the inverse-Hessian approximation using

           rho_k = 1 / (yk.T sk)

           H{k+1} = (I - rho_k sk yk.T) Hk
                      (I - rho_k yk sk.T) + rho_k sk sk.T.

The line search rejects geometrically invalid trials, including candidates
behind a camera. If numerical roundoff produces unsuitable BFGS curvature,
H is reset to the identity rather than allowing an indefinite search matrix.
"""
def estimate_3d_point_tensor(
    X0,
    image_points,
    states,
    Ks,
    Sigma_poses,
    Sigma_pixels,
    *,
    max_iterations=100,
    gradient_tolerance=1e-8,
    step_tolerance=1e-10,
    armijo_constant=1e-4,
    backtracking_factor=0.5,
    maximum_line_search_steps=25,
    return_info=False,
):
    """
    Refine an initial NED landmark estimate using full BFGS optimization.

    Args:
        X0: (3,) initial NED landmark estimate.
        image_points: (N,2) observed pixel coordinates.
        states: (N,7) camera states [position, quaternion], where each
            quaternion is ordered [w,x,y,z] and rotates body FRD -> NED.
        Ks: (N,3,3) camera intrinsic matrices.
        Sigma_poses: (N,6,6) camera pose covariance matrices.
        Sigma_pixels: (N,2,2) pixel measurement covariance matrices.
        max_iterations: Maximum number of accepted BFGS steps.
        gradient_tolerance: Stop when ||dF/dX||_infinity is below this.
        step_tolerance: Stop when the accepted position change is small.
        armijo_constant: Sufficient-decrease constant for line search.
        backtracking_factor: Multiplier applied when shortening a step.
        maximum_line_search_steps: Maximum trial steps per iteration.
        return_info: If True, return (X, information) instead of only X.

    Returns:
        The optimized (3,) tensor. It is detached from the temporary
        optimization graph. If return_info=True, it is accompanied by a
        dictionary containing convergence information and loss history.

    Notes:
        All numerical inputs are converted to X0's dtype and device. Float64
        is recommended because BFGS uses small differences of gradients.
    """
    # SECTION 1: VALIDATE OPTIMIZER SETTINGS
    # These ranges are required for a shrinking Armijo line search.
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must be between 0 and 1")
    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must be between 0 and 1")
    if max_iterations < 1 or maximum_line_search_steps < 1:
        raise ValueError("iteration limits must be positive")

    # SECTION 2: CREATE A PRIVATE OPTIMIZATION VARIABLE
    # Detaching and cloning prevents BFGS updates from altering the caller's X0
    # or retaining a computation graph from an earlier calculation.
    if isinstance(X0, torch.Tensor):
        if not torch.is_floating_point(X0):
            raise TypeError("X0 must be floating point")
        X = X0.detach().clone()
    else:
        X = torch.as_tensor(X0, dtype=torch.float64)

    if X.shape != (3,):
        raise ValueError("X0 must have shape (3,)")
    if not torch.isfinite(X).all():
        raise ValueError("X0 must contain only finite values")

    # SECTION 3: PREPARE CONSTANT MULTI-CAMERA DATA
    # Convert constant inputs once rather than once per objective evaluation.
    image_points = torch.as_tensor(
        image_points, dtype=X.dtype, device=X.device
    )
    states = torch.as_tensor(states, dtype=X.dtype, device=X.device)
    Ks = torch.as_tensor(Ks, dtype=X.dtype, device=X.device)
    Sigma_poses = torch.as_tensor(
        Sigma_poses, dtype=X.dtype, device=X.device
    )
    Sigma_pixels = torch.as_tensor(
        Sigma_pixels, dtype=X.dtype, device=X.device
    )

    # SECTION 4: DEFINE A REUSABLE VALUE-AND-GRADIENT EVALUATION
    # Each evaluation creates a fresh leaf tensor, evaluates F, asks autograd
    # for dF/dX, and detaches both outputs before the manual BFGS update.
    def value_and_gradient(point):
        point = point.detach().requires_grad_(True)
        value = maximum_likelihood_objective_tensor(
            point, image_points, states, Ks, Sigma_poses, Sigma_pixels
        )
        gradient, = torch.autograd.grad(value, point)
        if not torch.isfinite(value) or not torch.isfinite(gradient).all():
            raise ValueError("Objective or gradient became non-finite")
        return value.detach(), gradient.detach()

    # SECTION 5: INITIALIZE BFGS AT THE TRIANGULATED ESTIMATE
    # X0 is supplied by triangulation and H0 is the 3x3 identity matrix.
    H = torch.eye(3, dtype=X.dtype, device=X.device)
    value, gradient = value_and_gradient(X)
    loss_history = [value.item()]
    converged = False
    status = "maximum iterations reached"
    completed_iterations = 0

    for iteration in range(max_iterations):
        # 2) value_and_gradient uses autograd to calculate F(Xk) and dF/dX.
        gradient_norm = torch.linalg.vector_norm(gradient, ord=float("inf"))
        if gradient_norm.item() <= gradient_tolerance:
            converged = True
            status = "gradient tolerance reached"
            break

        # 3) Calculate the BFGS search direction pk = -Hk gk. If accumulated
        # roundoff makes this a non-descent direction, safely restart Hk.
        direction = -(H @ gradient)
        directional_derivative = gradient @ direction
        if directional_derivative.item() >= 0.0:
            H = torch.eye(3, dtype=X.dtype, device=X.device)
            direction = -gradient
            directional_derivative = -(gradient @ gradient)

        # 4) Backtracking line search enforces the Armijo decrease condition.
        # Invalid trials (behind a camera or invalid covariance) are rejected.
        step_size = 1.0
        accepted = False
        candidate_value = None
        for _ in range(maximum_line_search_steps):
            candidate = X + step_size * direction
            try:
                trial_value = maximum_likelihood_objective_tensor(
                    candidate, image_points, states, Ks,
                    Sigma_poses, Sigma_pixels
                ).detach()
                sufficient_decrease = (
                    value + armijo_constant * step_size
                    * directional_derivative
                )
                if (
                    torch.isfinite(trial_value)
                    and trial_value.item() <= sufficient_decrease.item()
                ):
                    candidate_value = trial_value
                    accepted = True
                    break
            except (ValueError, RuntimeError):
                pass
            step_size *= backtracking_factor

        if not accepted:
            status = "line search could not find a valid decreasing step"
            break

        # 5) Accept X{k+1} = Xk + alpha_k p_k and evaluate its gradient.
        X_next = candidate.detach()
        candidate_value, gradient_next = value_and_gradient(X_next)
        step = X_next - X
        gradient_change = gradient_next - gradient
        completed_iterations = iteration + 1
        loss_history.append(candidate_value.item())

        relative_step_limit = step_tolerance * (
            1.0 + torch.linalg.vector_norm(X).item()
        )

        # 6) Apply the inverse-Hessian BFGS update when yk.T sk has enough
        # positive curvature. Otherwise restart H to keep it positive definite.
        curvature = gradient_change @ step
        curvature_scale = (
            torch.linalg.vector_norm(gradient_change)
            * torch.linalg.vector_norm(step)
        )
        if curvature.item() > torch.finfo(X.dtype).eps * curvature_scale.item():
            rho = 1.0 / curvature
            identity = torch.eye(3, dtype=X.dtype, device=X.device)
            left = identity - rho * torch.outer(step, gradient_change)
            right = identity - rho * torch.outer(gradient_change, step)
            H = left @ H @ right + rho * torch.outer(step, step)
        else:
            H = torch.eye(3, dtype=X.dtype, device=X.device)

        X = X_next
        value = candidate_value
        gradient = gradient_next

        if torch.linalg.vector_norm(step).item() <= relative_step_limit:
            converged = True
            status = "step tolerance reached"
            break

    # SECTION 7: PACKAGE THE OPTIMUM AND DIAGNOSTICS
    # The returned estimate is detached because BFGS itself is an iterative
    # numerical procedure; callers normally differentiate F, not the optimizer.
    result = X.detach()
    information = {
        "converged": converged,
        "status": status,
        "iterations": completed_iterations,
        "objective": value.item(),
        "gradient": gradient.detach(),
        "gradient_norm": torch.linalg.vector_norm(
            gradient, ord=float("inf")
        ).item(),
        "loss_history": loss_history,
    }

    if return_info:
        return result, information
    return result


def estimate_3d_point(
    X0, image_points, states, Ks, Sigma_poses, Sigma_pixels, **kwargs
):
    """Run the tensor BFGS optimizer through the original public name."""
    return estimate_3d_point_tensor(
        X0, image_points, states, Ks, Sigma_poses, Sigma_pixels, **kwargs
    )
