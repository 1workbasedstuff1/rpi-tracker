the code handles the fusion of typically 2 or 3 camera position measurements combined with 
IMU + GNSS data which has the form 
Nominal State: x = [p, q]
Error State: δx = [δp, δθ]

and 6x6 error matrix of 
[Pδpδp, Pδpδθ]
[Pδθδp, Pδθδθ]

which is used in the camera estimation process

because position and pose is reported as a FRD relative to NED
(Local -> Global) we first have to transform the camera coordinate 
frame into the IMU's coordinate frame for 3D position tracking of 
objects

Camera has:
z - facing forward
x - pointing right
y - pointing down (assuming this is what opencv uses)

IMU has: 
z - down 
x - forward
y - right

Body FRD -> Camera Frame
R = 
[0, 1, 0]
[0, 0, 1]
[1, 0, 0]

Now lets say we have position and quaternion reported of the camera from IMU + GNSS
This transforms the global ned position into the cameras coordinate frame

X_camera = R_{bodyFRD -> Camera} R^-1_{local -> global} (X_ned - p)

Projection into Image plane

π(K X_camera)
where:
π (X,Y,Z) = (X/Z, Y/Z)
K - the camera's intrinsic callibration matrix


How do we Triangulate a 3d position from 2 or more camera positions?

Our observation model for each camera is 
h(X, Ki, ci, Ri) = π(K X_camera) (as shown above)

our estimation error would be:
ri = zi - hi 

zi - measured 2d pixel coordinate of object
hi - estimated 2 pixel coordinate based on estimation of 3d position X

Image Estimation noise:
Our image estimation noise/covariance has 2 components

S = S_pixel + S_pose
S_pixel: estimation noise of image sensor/ object tracker
S_pose: Estimation noise in image plane caused by pose errors
    S_pose =  ∂h/∂δξ Σ ∂h/∂δξ^T
    Σ - 6x6 matrix representing error matrix of δp,δθ
    δξ - [δp,δθ]
    ∂h/∂δξ - is a 2x6 matrix


In order to statistically optimal X we must minimise the quantity

X* = arg (min X) Σ (ri(X)^T Sj(X)^-1 ri(X) + log det (Sj(X)))

since this is NON-LINEAR optimisation we have opted to use 
BFGS optimise which needs an intial estimate


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



