"""
Helper functions for some 2D and 3D rotations. 

Some of the code was adapted from the following repos:
- https://github.com/akar43/lsm
"""
import pdb

import numpy as np
from IPython import embed

import scipy.misc
import torch
from scipy.linalg import logm, norm


def _detach(x):
    return x.detach().cpu().numpy()


def extract_viewpoint(vp, rot_rep):
    if rot_rep == "quat":
        vp = vp
    elif rot_rep == "azim":
        vp = vp.argmax(dim=1)
    elif rot_rep == "azim10":
        vp = vp.argmax(dim=1) * 10
    elif rot_rep == "azel":
        _az = vp[:, :360].argmax(dim=1, keepdim=True) - 180
        _el = vp[:, 360:].argmax(dim=1, keepdim=True) - 90
        vp = torch.cat((_az, _el), dim=1)
    elif rot_rep == "azel10":
        _az = vp[:, :36].argmax(dim=1, keepdim=True) - 18
        _el = vp[:, 36:].argmax(dim=1, keepdim=True) - 9
        vp = torch.cat((_az, _el), dim=1) * 10
    elif rot_rep == "euler":
        _e1 = vp[:, :360].argmax(dim=1, keepdim=True) - 180
        _e2 = vp[:, 360:540].argmax(dim=1, keepdim=True) - 90
        _e3 = vp[:, 540:].argmax(dim=1, keepdim=True) - 180
        vp = torch.cat((_e1, _e2, _e3), dim=1)
    elif rot_rep == "euler60":
        _e1 = vp[:, :6].argmax(dim=1, keepdim=True) - 6
        _e2 = vp[:, 6:9].argmax(dim=1, keepdim=True) - 1.5
        _e3 = vp[:, 9:16].argmax(dim=1, keepdim=True) - 6
        vp = torch.cat((_e1, _e2, _e3), dim=1) * 60
    elif rot_rep == "euler10":
        _e1 = vp[:, :36].argmax(dim=1, keepdim=True) - 18
        _e2 = vp[:, 36:54].argmax(dim=1, keepdim=True) - 9
        _e3 = vp[:, 54:].argmax(dim=1, keepdim=True) - 18
        vp = torch.cat((_e1, _e2, _e3), dim=1) * 10
    elif rot_rep == "matrix":
        vp = vp
    return vp


def theta_distance(pr, gt, theta_range):
    error = (gt - pr).abs().detach().cpu().numpy()[:, None]
    error = np.concatenate((error, theta_range - error), axis=1)
    return error.min(axis=1)


##################################################
##### Utility functions for quaternions      #####
##################################################
_normalize_quat = lambda q: q * np.sign(q[0])
_Qx = lambda x: np.array([np.cos(x / 2), np.sin(x / 2), 0, 0])
_Qy = lambda x: np.array([np.cos(x / 2), 0, np.sin(x / 2), 0])
_Qz = lambda x: np.array([np.cos(x / 2), 0, 0, np.sin(x / 2)])


def euler2quat(E, order="XYZ", deg=False):
    if deg:
        E = E * np.pi / 180.0
    if (E > 7.0).any():
        raise ValueError("Radians not degreess: {}".format(E))
    Q = {"X": _Qx, "Y": _Qy, "Z": _Qz}
    Q1 = Q[order[0]](E[0])
    Q2 = Q[order[1]](E[1])
    Q3 = Q[order[2]](E[2])
    Q12 = hamilton_product(Q1, Q2)
    Q123 = hamilton_product(Q12, Q3)
    return Q123


def relative_quaternion(q_t, q_r):
    """
    Calculates difference quaternion between two quaternions
    q' = relative_quaternion(q1, q2) such that q' = angle(q2) - angle(q1) 
    https://stackoverflow.com/questions/22157435/difference-between-the-two-quaternions
    """
    q = hamilton_product(q_t, quat_conjugate(q_r))
    return _normalize_quat(q)


def quat_conjugate(q):
    q = np.array([q[0], -q[1], -q[2], -q[3]])
    return _normalize_quat(q)


def hamilton_product(q1, q2):
    # https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q = np.zeros_like(q1)
    q[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    q[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    q[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    q[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return _normalize_quat(q)


def rotate_point_quat(p, q):
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations
    assert len(p) == 3
    p_h = np.zeros(4)
    p_h[1:] = p
    pr_h = hamilton_product(hamilton_product(q, p_h), quat_conjugate(q))
    pr = pr_h[1:]
    return pr


##################################################
##### Utility function for rotation matrices #####
##################################################
def _Rx(x):
    return np.array(
        [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
    )


def _Ry(x):
    return np.array(
        [[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-np.sin(x), 0, np.cos(x)]]
    )


def _Rz(x):
    return np.array(
        [[np.cos(x), -np.sin(x), 0], [np.sin(x), np.cos(x), 0], [0, 0, 1]]
    )


def euler2matrix(E, order="XYZ", deg=False):
    assert len(E) == 3
    if deg:
        E = E * np.pi / 180.0
    if (E > 7.0).any():
        raise ValueError("Radians not degreess: {}".format(E))
    R = {"X": _Rx, "Y": _Ry, "Z": _Rz}
    return R[order[2]](E[2]) @ R[order[1]](E[1]) @ R[order[0]](E[0])


def matrix2euler(R, order):
    # https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    # matlabs rotm2eul implementation
    if order == "XYZ":
        i, j, k = 0, 1, 2
        parity = False
    elif order == "ZYX":
        i, j, k = 2, 1, 0
        parity = True

    sy = np.sqrt(R[i, i] * R[i, i] + R[j, i] * R[j, i])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[k, j], R[k, k])
        y = np.arctan2(-R[k, i], sy)
        z = np.arctan2(R[j, i], R[i, i])
    else:
        x = np.arctan2(-R[j, k], R[j, j])
        y = np.arctan2(-R[k, i], sy)
        z = 0

    eul = np.array([x, y, z])

    if parity:
        eul = -1 * eul

    return eul


def azel2matrix(E, order="XYZ"):
    """
    Azimuth and Elevation are the rotations about Z and Y in this repo. 
    """
    if (E > 7.0).any():
        raise ValueError("Radians not degreess: {}".format(E))
    E = list(E)
    E.append(0)
    return euler2matrix(np.array(E), order=order)


def rotate_point_euler(p, E):
    R = euler2matrix(E)
    return np.matmul(R, p)


def matrix_angle(R):
    angle = (1.0 / np.sqrt(2)) * norm(logm(R, False)[0], "fro")
    return angle


#########################
# -- Error Calculations
#########################
def relative_theta(tar, ref, period):
    rel = (tar - theta) % period
    return rel


def relative_matrix(tar, ref):
    inv_ref = ref.t() if type(ref) is torch.Tensor else ref.T
    return tar @ inv_ref


def relative_euler(tar, ref):
    R_t = euler2matrix(tar)
    R_r = euler2matrix(ref)
    R = relative_matrix(R_t, R_r)
    return matrix2euler(R)


def euler2geodist(pr, gt):
    # convert to radians
    pr = pr.float() * np.pi / 180.0
    gt = gt.float() * np.pi / 180.0
    pr = _detach(pr)
    gt = _detach(gt)

    def _dist_instance(_pr, _gt):
        predR = euler2matrix(_pr)
        gtR = euler2matrix(_gt)
        return matrix_angle(np.matmul(predR.T, gtR)) * 180.0 / np.pi

    errors = np.zeros(pr.shape[0], dtype=np.float)
    for i in range(pr.shape[0]):
        errors[i] = _dist_instance(pr[i], gt[i])
    return errors


def azel2geodist(pr, gt, order="XYZ"):
    # convert to radians
    pr = pr.float() * np.pi / 180.0
    gt = gt.float() * np.pi / 180.0
    pr = _detach(pr)
    gt = _detach(gt)

    def _dist_instance(_pr, _gt):
        predR = azel2matrix(_pr, order)
        gtR = azel2matrix(_gt, order)
        return matrix_angle(np.matmul(predR.T, gtR)) * 180.0 / np.pi

    errors = np.zeros(pr.shape[0], dtype=np.float)
    for i in range(pr.shape[0]):
        errors[i] = _dist_instance(pr[i], gt[i])
    return errors


def matrix2geodist(pr, gt):
    # convert to radians
    pr = _detach(pr.float())
    gt = _detach(gt.float())

    def _dist_instance(_pr, _gt):
        return matrix_angle(np.matmul(_pr.T, _gt)) * 180.0 / np.pi

    errors = np.zeros(pr.shape[0], dtype=np.float)
    for i in range(pr.shape[0]):
        errors[i] = _dist_instance(pr[i], gt[i])
    return errors


def quaternion_distance(pred, gt):
    rel_w = (pred * gt).sum(dim=1).abs()
    return 2 * rel_w.acos()


def get_vp_error(vp_pr, vp_gt, rot_rep, embed=False):
    if rot_rep == "quat":
        vp_error = quaternion_distance(vp_pr, vp_gt) * 180.0 / np.pi
        vp_error = _detach(vp_error)
    elif "azim" in rot_rep:
        vp_gt = vp_gt - 180.0
        vp_error = theta_distance(vp_pr, vp_gt, 360.0)
    elif "azel" in rot_rep:
        vp_gt = vp_gt - torch.LongTensor([[180, 90]]).cuda()
        vp_error = azel2geodist(vp_pr, vp_gt)
    elif "euler" in rot_rep:
        vp_gt = vp_gt - torch.FloatTensor([[180, 90, 180]]).cuda()
        vp_error = euler2geodist(vp_pr, vp_gt)
    elif "R" == rot_rep:
        vp_error = matrix2geodist(vp_pr, vp_gt)
    elif "Rt" in rot_rep:
        vp_error = matrix2geodist(vp_pr[:, :, :3], vp_gt[:, :, :3])
    elif rot_rep == "binary":
        vp_error = 1 - (vp_gt.byte() == (vp_pr >= 0.5).byte()).float().mean(
            dim=1
        )
        vp_error = _detach(vp_error)

    return vp_error


if __name__ == "__main__":
    embed()
