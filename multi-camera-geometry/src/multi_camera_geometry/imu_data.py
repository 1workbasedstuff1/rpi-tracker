import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import multi_camera_geometry.helper as hp

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
def ned2cameraframe(x_ned, state, K, delta):
    """
    x_ned: (3,)    Landmark in NED
    state: (7,)    [p, q], quaternion (w,x,y,z)
    K:     (3,3)   Camera intrinsics
    delta: (6,)    [delta_p, delta_theta]

    Returns:
        (2,) Image coordinates in pixels
    """

    p = state[:3]
    q = F.normalize(state[3:7], dim=-1)

    delta_p = delta[:3]
    delta_theta = delta[3:6]

    # Perturbed position
    p_perturbed = p + delta_p

    # Transform NED point into body FRD frame
    X_body = hp.quaternion_rotate(
        hp.quaternion_conjugate(q),
        x_ned - p_perturbed
    )

    # Small-angle quaternion error, right-multiplicative
    delta_q = torch.cat([
        torch.ones_like(delta_theta[:1]),
        0.5 * delta_theta
    ])

    delta_q = F.normalize(delta_q, dim=-1)

    # Apply inverse orientation error
    X_body = hp.quaternion_rotate(
        hp.quaternion_conjugate(delta_q),
        X_body
    )

    # Body FRD -> Camera
    R_bc = state.new_tensor([
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.]
    ])

    X_camera = R_bc @ X_body

    # Pixel projection
    projected = K @ X_camera

    return projected[:2] / projected[2]


