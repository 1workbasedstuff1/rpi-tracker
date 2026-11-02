import torch
import torch.nn.functional as F
import numpy as np

def quaternion_rotate(q, v):
    """
    Rotate a 3D vector by a quaternion.

    q: (..., 4) quaternion in (w, x, y, z) order
    v: (..., 3) vector

    Returns: (..., 3) rotated vector
    """
    q = F.normalize(q, dim=-1)

    w = q[..., :1]
    xyz = q[..., 1:]

    t = 2.0 * torch.cross(xyz, v, dim=-1)

    return v + w * t + torch.cross(xyz, t, dim=-1)

def project_π(X_c):
    """
    Project 3D camera coordinates to normalized image coordinates.

    X_c: (..., 3)
    Returns: (..., 2)
    """
    return X_c[..., :2] / X_c[..., 2:3]

def quaternion_conjugate(q):
    return torch.cat([q[:1], -q[1:]])

def skew(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def quaternion_rotate_tensor(q, v):
    """
    Rotate vectors with quaternions while preserving the autograd graph.

    Args:
        q: (..., 4) quaternion tensor in [w, x, y, z] order.
        v: (..., 3) vector tensor with batch dimensions compatible with q.

    Returns:
        (..., 3) tensor containing q v q-conjugate.
    """
    # Unit quaternions represent pure rotations. Normalizing here also makes
    # the function robust to small normalization errors in estimated poses.
    q = F.normalize(q, dim=-1)

    # Split q into its scalar component w and vector component [x, y, z].
    w = q[..., :1]
    xyz = q[..., 1:]

    # This is the efficient quaternion-vector rotation identity:
    #     t = 2(q_xyz cross v)
    #     v_rotated = v + w t + q_xyz cross t
    # It avoids explicitly constructing q * [0,v] * conjugate(q).
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def project_π_tensor(X_c):
    """
    Apply perspective division to homogeneous camera/image coordinates.

    Args:
        X_c: (..., 3) tensor whose final coordinate is the projective scale.

    Returns:
        (..., 2) tensor [X_c[...,0]/X_c[...,2],
        X_c[...,1]/X_c[...,2]].

    The caller must ensure the denominator is positive and not close to zero.
    """
    # Keep the last dimension so broadcasting works for individual points and
    # batches without creating any detached scalar tensors.
    return X_c[..., :2] / X_c[..., 2:3]


def quaternion_conjugate_tensor(q):
    """
    Return the conjugate of quaternions in [w, x, y, z] order.

    For a unit quaternion, the conjugate is also its inverse. Therefore this
    is used to reverse a body-to-NED rotation into a NED-to-body rotation.
    """
    # Preserve the scalar part and negate all three vector components.
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def skew_tensor(v):
    """
    Construct the skew-symmetric cross-product matrix of one 3D vector.

    For any vector u, skew_tensor(v) @ u is equal to v cross u. Building the
    matrix with torch.stack preserves gradients with respect to v.
    """
    if v.shape != (3,):
        raise ValueError("v must be a 3D vector")

    # Unbind retains each component in the computation graph.
    x, y, z = v.unbind()
    zero = torch.zeros_like(x)

    # Do not use torch.tensor([[0, -z, y], ...]); that would copy and detach
    # x, y, and z, preventing gradients from flowing through this matrix.
    return torch.stack([
        torch.stack([zero, -z, y]),
        torch.stack([z, zero, -x]),
        torch.stack([-y, x, zero]),
    ])


def quaternion_to_rotation_matrix_tensor(q):
    """
    Convert [w, x, y, z] quaternions into 3x3 rotation matrices.

    Args:
        q: (..., 4) quaternion tensor.

    Returns:
        (..., 3, 3) rotation matrices that rotate in the same direction as
        quaternion_rotate_tensor().
    """
    # Normalize along the quaternion dimension so the resulting matrix is an
    # orthonormal rotation matrix rather than a scaled transformation.
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)

    # Evaluate the standard quaternion-to-matrix formula component by
    # component. torch.stack retains gradients with respect to q.
    matrix_components = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=-1)

    # Restore the final two matrix dimensions while keeping any leading batch
    # dimensions supplied by the caller.
    return matrix_components.reshape(q.shape[:-1] + (3, 3))
