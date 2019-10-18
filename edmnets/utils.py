import numpy as np
import edmnets.hungarian._binding as _bd


def reorder_distance_matrices(D1, D2, cost=None, types1=None, types2=None):
    """
    Reorders a batch of distance matrices to match another batch as distance matrices as closely
    as possible (element-wise). To this end, a cost matrix can either be constructed or passed into the function.
    If constructed, it is computed as absolute difference in mean distance from each single atom to all others in its
    point cloud. Furthermore, types can be passed which are used to restrict the matching to only certain permutation
    groups (the ones with matching types).

    :param D1: first distance matrix, shape [B, N, N]
    :param D2: second distance matrix, shape [B, N, N]
    :param cost: cost matrix, optional
    :param types1: sparse types corresponding to first distance matrix, shape [B, N]
    :param types2: sparse types corresponding to second distance matrix, shape [B, N]
    :return: a reordered version of D1 and types1 to match D2 and types2 as closely as possible
    """
    if types1 is None:
        types1 = np.zeros((D1.shape[0], D1.shape[1]), dtype=np.int32)
    if types2 is None:
        types2 = np.zeros((D1.shape[0], D1.shape[1]), dtype=np.int32)
    if cost is None:
        cost = np.abs(np.mean(D1, axis=1)[..., None] - np.mean(D2, axis=1)[:, None, :])
    out = _bd.hungarian_reorder(np.copy(cost), np.copy(D1), np.copy(types1), types2)
    return out[0].astype(D2.dtype), out[1]


def to_T_matrix(D):
    """
    implements the operation -J D J/2, D is EDM iff the result is positive semi-definite
    """
    n_atoms = D.shape[0]
    J = np.eye(n_atoms) - np.ones((n_atoms, n_atoms)) / float(n_atoms)
    return -0.5 * J @ D @ J


def is_edm(mats: np.ndarray, atol: float = 1e-5, squared: bool = True):
    """
    Checks whether an array of matrices are euclidean distance matrices.
    :param mats: array of matrices, expected shape [B, N, N] where B is number of matrices,
                 N number of represented points
    :param atol: tolerance with which to check
    :param squared: whether the matrices are indeed EDMs or contain actual pairwise distances
    :return: a boolean array indicating EDM-ness for each matrix
    """
    assert len(mats.shape) == 3 and mats.shape[1] == mats.shape[2], \
        f"Need distance matrix shape to be [B, N, N] but was {mats.shape}"
    if not squared:
        mats = np.power(mats, 2.)
    out = np.empty((len(mats),), dtype=np.bool)
    out[:] = False
    for mix in range(len(mats)):
        mat = mats[mix]
        T = to_T_matrix(mat)
        evs = np.linalg.eigvalsh(T)
        out[mix] = np.all(evs >= -atol)
    return out


def to_distance_matrices(coordinates: np.ndarray, squared: bool = True):
    """
    Converts a list of points to a list of corresponding euclidean distance matrices
    :param coordinates: coordinates array, expected shape [B, N, D] where B is number of point clouds,
                        N is number of points, D is dimension of points
    :param squared: whether to return squared distances (EDMs) or pairwise Euclidean distances
    :return: the output array of distance matrices
    """
    assert len(coordinates.shape) == 3
    ri = np.expand_dims(coordinates, axis=2)
    rj = np.expand_dims(coordinates, axis=1)
    rij = ri - rj
    output = np.add.reduce(np.square(rij), axis=-1, keepdims=False)
    if not squared:
        output = np.sqrt(output)
    return output


def to_M_matrix(D):
    """
    Converts an EDM to its Gram matrix representation with M_ij = <x_i - x_1, x_j - x_1>.
    :param D: the EDM
    :return: the Gram matrix
    """
    D1j = np.tile(D[0, :], len(D)).reshape(D.shape)
    Di1 = D1j.T
    return .5 * (-D + D1j + Di1)


def to_D_matrix(M):
    """
    Converts a Gram matrix M corresponding to a point cloud to an EDM D.
    :param M: The Gram matrix
    :return: a Euclidean distance matrix D
    """
    Mii = np.diag(M)
    return np.expand_dims(Mii, 0) + np.expand_dims(Mii, 1) - 2. * M


def to_coordinates(distance_matrices: np.ndarray, squared: bool = True, ndim: int = 3):
    """
    Converts a list of EDMs to Cartesian coordinates. This operation is unique up to rotation, translation, and
    mirroring of the resulting point clouds.

    :param distance_matrices: an array of EDMs, expected shape [B, N, N] where B is the number of EDMs and N
                              the number of points
    :param squared: whether the input consists out of EDMs or pairwise distances
    :param ndim: the embedding dimension
    :return: a list of Cartesian coordinates, shape [B, N, ndim]
    """
    assert len(distance_matrices.shape) == 3 and distance_matrices.shape[1] == distance_matrices.shape[2], \
        f"Need distance matrix shape to be [B, N, N] but was {distance_matrices.shape}"
    if not squared:
        distance_matrices = np.power(distance_matrices, 2)
    coordinates = np.empty((distance_matrices.shape[0], distance_matrices.shape[1], ndim),
                           dtype=distance_matrices.dtype)
    for batch_ix in range(len(distance_matrices)):
        D = distance_matrices[batch_ix]

        M = to_M_matrix(D)
        S, U = np.linalg.eigh(M)

        coord = np.matmul(U, np.diag(np.sqrt(np.abs(S))))
        coord = coord[:, -ndim:]
        coordinates[batch_ix] = coord
    return coordinates
