"""
Trial covariance matrix rotations.
Rotate the position 3x3 and velocity 3x3 by the same rotation,
and also the covariance 3x3. See if consistent

Implented using Tait-Bryan angles (intrinsic)

Struggling to visualise how the the matrix operations correspond to the
angles of rotation. Maybe plot some vectors and see how they transform...?

Helpful (maybe) stack overflow answer
https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance

Helpful link on SO3 parametrisations
https://en.wikipedia.org/wiki/Charts_on_SO(3)
"""
import numpy as np

np.set_printoptions(suppress=True)

def r_x(theta):
    result = np.array([
        [1., 0., 0.,],
        [0., np.cos(theta), -np.sin(theta),],
        [0., np.sin(theta),  np.cos(theta),],
    ])
    return result

def r_y(theta):
    result = np.array([
        [np.cos(theta), 0., np.sin(theta),],
        [0., 1., 0.,],
        [-np.sin(theta), 0., np.cos(theta),],
    ])
    return result

def r_z(theta):
    result = np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.,],
        [0., 0., 1.,],
    ])
    return result


def genRotationMatrix(alpha=0.0, beta=0.0, gamma=0.0):
    """
    Generate a rotation matrix based on 3 angles

    Implemented based on coordinate system of the rigid body. Imagine a plane
    being pointed at a cardinal direction (NS,EW) by amount `alpha`,
    being tilted up or down to a 'height' by amount `beta`, then rolled
    on its axis by amount `gamma`.

    In order to conserve rotations performed on XYZ across to those performed
    on UVW, the 6x6 rotation matrix is composed of an identical 3x3 rotation
    matrix in the top left and bottom right corners, with zeros everywhere else.

    Parameters
    ----------
    alpha : degree
        angle about Z axis (yaw), angle covers a span of (0,360)
    beta : degree
        angle about Y axis (pitch), angle covers a span of (-90,90)
    gamma : degree
        angle about X axis (roll), angle covers a span of (0,360)
    """
    sub_rotation_matrix = np.dot(r_z(alpha), np.dot(r_y(beta), r_x(gamma)))
    empty_block = np.zeros((3,3))

    rot_mat = np.block([[sub_rotation_matrix, empty_block],
                        [empty_block, sub_rotation_matrix]])

    return rot_mat


def rotateMatrix2(cov_mat, alpha=0.0, beta=0.0, gamma=0.0):
    """
    Rotate a matrix

    R = R_z(alpha) R_y(beta) R_x(gamma)
    A rotation with yaw, pitch, roll

    Parameters
    ----------
    cov_mat : (6x6) array
        initial covariance matrix
    alpha : degree
        (yaw) (0,360)
    beta : degree
        (pitch) (-90,90)
    gamma : degree
        (roll) (0,360)
    """
    rotation_matrix = np.dot(r_z(alpha), np.dot(r_y(beta), r_x(gamma)))
    pos_res = np.dot(rotation_matrix, np.dot(cov_mat[:3,:3], rotation_matrix.T))
    vel_res = np.dot(rotation_matrix, np.dot(cov_mat[3:,3:], rotation_matrix.T))
    corr_res = np.dot(rotation_matrix, np.dot(cov_mat[:3,3:], rotation_matrix.T)) # upper right quarter

    comb_res = np.vstack(
        (np.hstack((pos_res, corr_res)),
        np.hstack((corr_res.T, vel_res))),
    )

    return comb_res

def rotateMatrix(cov_mat, alpha=0., beta=0., gamma=0.):
    """
    Rotate a 6x6 covariance matrix based on 3 angles.

    Assumes the covariance is cartesian, with values corresponding to
    XYZUVW. Rotation only requires 3 angles because XYZ must be rotated
    identically to UVW.

    Implemented based on coordinate system of the rigid body. Imagine a plane
    being pointed at a cardinal direction (NS,EW) by amount `alpha`,
    being tilted up or down to a 'height' by amount `beta`, then rolled
    on its axis by amount `gamma`.

    Parameters
    ----------
    cov_mat : (6x6) numpy array
        covariance matrix in cartesian phase-space XYZUVW
    alpha : degree
        angle about Z axis (yaw), angle covers a span of (0,360)
    beta : degree
        angle about Y axis (pitch), angle covers a span of (-90,90)
    gamma : degree
        angle about X axis (roll), angle covers a span of (0,360)
    """
    rot_mat = genRotationMatrix(alpha, beta, gamma)
    return np.dot(rot_mat, np.dot(cov_mat, rot_mat.T))

M = 2.0
N = 1.0
P = 1.0
Q = 2.0
corr_XV = 0.9
COV_MAT = np.zeros((6,6))
COV_MAT[0,0] = M**2
COV_MAT[1,1] = N**2
COV_MAT[2,2] = N**2
COV_MAT[3,3] = P**2
COV_MAT[4,4] = Q**2
COV_MAT[5,5] = P**2
COV_MAT[0,4] = M * P * corr_XV
COV_MAT[4,0] = M * P * corr_XV

def test_ZeroAngle():
    """Test that zero angles don't modify result"""
    result = rotateMatrix(COV_MAT)
    assert np.allclose(COV_MAT, result, atol=1e-5)

def test_90AngleX():
    """Test that 90 rotation about X behaves as expected

    Y and Z should swap along diagonal, as should V and W.
    The XV correlation should move to XW
    """
    result = rotateMatrix(COV_MAT, gamma=np.pi/2)

    expected_result = np.copy(COV_MAT)
    expected_result[1,1], expected_result[2,2] =\
        expected_result[2,2], expected_result[1,1]
    expected_result[4,4], expected_result[5,5] = \
        expected_result[5,5], expected_result[4,4]

    expected_result[0,4], expected_result[0,5] =\
        expected_result[0,5], expected_result[0,4]
    expected_result[4,0], expected_result[5,0] = \
        expected_result[5,0], expected_result[4,0]
    #
    # expected_result[]
    #
    # expected_result = np.zeros((6,6))
    # expected_result[:,0] = COV_MAT[:,0]
    # expected_result[:,1] = COV_MAT[:,2]
    # expected_result[:,2] = COV_MAT[:,1]
    # expected_result[:,3] = COV_MAT[:,3]
    # expected_result[:,4] = COV_MAT[:,5]
    # expected_result[:,5] = COV_MAT[:,4]
    # expected_result[:3,3:] *= -1
    # expected_result[3:,:3] *= -1

    print(expected_result)
    print(result)
    assert np.allclose(expected_result, result, atol=1e-5)


def test_combinedAngles():
    """
    Iteratively put `COV_MAT` through rotations, confirming stds to be unity
    or not as appropriate.

    Note that `COV_MAT` is initially 'fat' in X and V.

    TODO: check that things 'point' in the right direction, maybe need vectors
    """

    # First perform a rotation about Z (and W) of 90 degrees
    rot_Z = rotateMatrix(COV_MAT, alpha=np.pi/2)

    # should now be fat in Y and U with all else being unit
    diag = np.diagonal(rot_Z)
    assert np.allclose(1., diag[np.array([0,2,4,5])])
    assert np.all(np.logical_not(np.isclose(1., diag[np.array([1,3])])))

    # Now repeat but add rotation about Y (and V) of 90 degrees
    rot_Z_and_Y = rotateMatrix(COV_MAT, alpha=np.pi/2, beta=np.pi/2)

    # should now be fat in Z and U
    diag = np.diagonal(rot_Z_and_Y)
    assert np.allclose(1., diag[np.array([0,1,4,5,])])
    assert np.all(np.logical_not(np.isclose(1., diag[np.array([2,3])])))

    # Now repeat but add rotation about X (and U) of 90 degrees
    rot_Z_Y_and_X = rotateMatrix(COV_MAT,
                                 alpha=np.pi/2,
                                 beta=np.pi/2,
                                 gamma=np.pi/2)

    # should now be fat in Z and V
    diag = np.diagonal(rot_Z_Y_and_X)
    assert np.allclose(1., diag[np.array([0,1,3,5,])])
    assert np.all(np.logical_not(np.isclose(1., diag[np.array([2,4])])))




if __name__ == '__main__':
    res = rotateMatrix(COV_MAT)
