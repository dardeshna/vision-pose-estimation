import re
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from math import cos, sin, radians
from numpy.core.shape_base import atleast_2d
import scipy.optimize

rng = np.random.default_rng()

rotation_matrix_2d = lambda theta : np.array([[cos(radians(theta)), -sin(radians(theta))], [sin(radians(theta)), cos(radians(theta))]])

def rotation_matrix_3d(yaw=0, pitch=0, roll=0):

    yaw = radians(yaw)
    pitch = radians(pitch)
    roll = radians(roll)

    Ry = np.array([
        [cos(yaw), 0, sin(yaw)],
        [0, 1, 0],
        [-sin(yaw), 0, cos(yaw)],
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, cos(pitch), -sin(pitch)],
        [0, sin(pitch), cos(pitch)],
    ])

    Rz = np.array([
        [cos(roll), -sin(roll), 0],
        [sin(roll), cos(roll), 0],
        [0, 0, 1],
    ])

    return Ry@Rx@Rz

cameraMatrix = np.array([
    [772.53876202, 0., 479.132337442],
    [0., 769.052151477, 359.143001808],
    [0., 0., 1.0],
])

cameraMatrix_inv = np.linalg.inv(cameraMatrix)

distortionCoefficient = np.array([2.9684613693070039e-01, -1.4380252254747885e+00,-2.2098421479494509e-03,
                        -3.3894563533907176e-03, 2.5344430354806740e+00])

focalLength = 2.9272781257541; #mm

# print(cameraMatrix)
# print(distortionCoefficient)
# print(focalLength)

target = np.zeros((8, 2))
target[0] = np.array([17, 0])
target[1] = rotation_matrix_2d(-60)@target[0]
target[2:4] = target[1::-1]*np.array([-1, 1])
target[4:8] = target[3::-1]*(target[1, 1]+2)/target[1, 1]
target = target*np.array([1, -1]) # flip y axis
target_moments = cv.moments(target.astype(np.float32))
target_centroid_y = target_moments['m01']/target_moments['m00']
target_height = 6*12+9.25+target[1, 1]-target_centroid_y
camera_height = 32

target_3d = np.vstack((target[:, 0], target[:, 1]-target_centroid_y, np.zeros(8), np.ones(8)))
panel_3d = np.vstack((np.array([[-20, -20, 20, 20, -20], [target_height, -30, -30, target_height, target_height]]), np.zeros(5), np.ones(5))) 

yaw_wc_truth = 10
pitch_wc_truth = 35
x_wc_truth = -3*12
z_wc_truth = -10*12

t_wc_truth = np.array([x_wc_truth, target_height-camera_height, z_wc_truth])
R_wc_truth = rotation_matrix_3d(yaw=yaw_wc_truth, pitch=pitch_wc_truth)

# t_wc_truth = np.array([0, 0, -10*12])
# R_wc_truth = rotation_matrix_3d(pitch=-20)

T_cw_truth = np.hstack((R_wc_truth.T, np.atleast_2d(-R_wc_truth.T@t_wc_truth).T))

target_c = cameraMatrix@T_cw_truth@target_3d
target_c = target_c/target_c[2]

panel_c = cameraMatrix@T_cw_truth@panel_3d
panel_c = panel_c/panel_c[2]

fig, ax = plt.subplots()

n = 100
alpha_centroid = np.zeros(n)
d_centroid = np.zeros(n)

alpha_true_centroid = np.zeros(n)
d_true_centroid = np.zeros(n)

t_wc_est = np.zeros((n, 3))
t_wc_est_true_contour = np.zeros((n, 3))

noise = 1

to_T = lambda R, t : np.vstack((np.hstack((R, np.atleast_2d(t).T)), np.array([[0, 0, 0, 1]])))

for i in range(n):

    target_c_n = target_c[0:2] + noise*rng.standard_normal(size=target_c[0:2].shape)
    target_c_n_moments = cv.moments(target_c_n.T.astype(np.float32))
    target_c_n_centroid = np.array([target_c_n_moments['m10'], target_c_n_moments['m01']])/target_c_n_moments['m00']

    target_c_n_H, retval = cv.findHomography(target_3d[0:2].T, target_c_n[0:2].T)
    target_c_n_true_centroid = target_c_n_H[0:2, 2]

    retval, r_cw_est, t_cw_est = cv.solvePnP(target_3d[0:3].T, target_c_n[0:2].T, cameraMatrix, None)
    R_cw_est, jac = cv.Rodrigues(r_cw_est)

    t_wc_est[i] = np.squeeze(-R_cw_est.T@t_cw_est)

    # plt.scatter(t_wc_est[0], t_wc_est[2])

    # print(t_cw_est, -R_wc_truth.T@t_wc_truth)

    # print(-R_cw_est.T@t_cw_est, t_wc_truth)

    intermediate = rotation_matrix_3d(pitch=pitch_wc_truth)@cameraMatrix_inv@np.append(target_c_n_centroid, 1)
    alpha_centroid[i] = alpha = np.degrees(np.arctan2(-intermediate[0], intermediate[2]))
    d_centroid[i] = d = -(target_height-camera_height)/intermediate[1]*np.sqrt(intermediate[0]**2+intermediate[2]**2)

    # print(alpha, d)

    intermediate = rotation_matrix_3d(pitch=pitch_wc_truth)@cameraMatrix_inv@np.append(target_c_n_true_centroid, 1)
    alpha_true_centroid[i] = alpha = np.degrees(np.arctan2(-intermediate[0], intermediate[2]))
    d_true_centroid[i] = d = -(target_height-camera_height)/intermediate[1]*np.sqrt(intermediate[0]**2+intermediate[2]**2)

    R_wc_prime_est = rotation_matrix_3d(yaw=alpha_true_centroid[i], pitch=pitch_wc_truth)
    T_cw_prime_est = np.hstack((R_wc_prime_est.T, np.atleast_2d(-R_wc_prime_est.T@np.array([0, target_height-camera_height, -d_true_centroid[i]])).T))

    convert_from_homogenous = lambda x : x[0:2]/x[2]

    reproject = lambda skew : convert_from_homogenous(
        cameraMatrix@T_cw_prime_est@np.append(
            rotation_matrix_3d(yaw=-skew)@target_3d[0:3], np.ones((1, target_3d.shape[-1])), axis=0))

    res = scipy.optimize.minimize(lambda skew: np.sum((target_c_n[0:2] - reproject(skew))**2), 0)

    t_wc_est_true_contour[i] = rotation_matrix_3d(yaw=res['x'])@np.array([0, target_height-camera_height, -d_true_centroid[i]])

    kh = cameraMatrix_inv@target_c_n_H

    print(np.dot(kh[:, 0], kh[:, 1]))

    # x1 = to_T(target_c_n_H, np.zeros(3))@to_T(np.eye(3), np.array([0,0,1]))
    # x2 = to_T(R_wc_prime_est.T, -R_wc_prime_est.T@np.array([0, target_height-camera_height, -d_true_centroid[i]]))

    # H_test = cameraMatrix@T_cw_truth
    # H_test = H_test[:, [0, 1, 3]]/H_test[2, 3]

    # print(H_test)
    # print(target_c_n_H)
    
    # print(T_cw_truth[:, [0, 1, 3]])
    # print(cameraMatrix_inv@H_test*T_cw_truth[2, 3])

    # U, S, Vh = np.linalg.svd(T_cw_truth[:, [0, 1, 3]])
    # W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # print(S)

    # U, S, Vh = np.linalg.svd(cameraMatrix_inv@H_test)
    # W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # print(S)

    # print(cv.Rodrigues(U@W@Vh)[0])
    # print(cv.Rodrigues(U@W.T@Vh)[0])



    
    # print(cv.Rodrigues(res[0])[0])
    # print(cv.Rodrigues(res[1])[0])

    # print(cv.Rodrigues(R_wc_truth.T)[0])



    # print(target_3d.shape)
    # target = (R_wc_truth.T@target_3d[0:3])[0:2].T
    # target = (R_wc_truth.T@target_3d[0:3])[0:2].T

    # for R, t, n in zip(res[1], res[2], res[3]):
        

    # print(np.linalg.det(H_R), np.linalg.det(H_Q), np.linalg.det(cameraMatrix))

    # print(x1)
    # print(x2)

    # print(np.linalg.det(x1[0:3, 0:3]))
    # print(np.linalg.det(x2))

    # print(np.linalg.det(cameraMatrix_inv@target_c_n_H))

    # x = np.linalg.inv(to_T(np.eye(3), -1*np.array([0, target_height-camera_height, -d_true_centroid[i]])))@to_T(R_wc_prime_est@cameraMatrix_inv@target_c_n_H, np.zeros(3))@to_T(np.eye(3), np.array([0,0,1]))
    # print(x)
    
    # print(np.linalg.eigvals(x[0:3, 0:3]))
    # print(np.linalg.eig(x[0:3, 0:3]))
    # print(np.linalg.det(x[0:3, 0:3]))
    
    
    # print(t_wc_truth)

    # reprojected_pts = reproject(res['x'])
    # print(alpha, d)

    # print(np.degrees(np.arctan2(t_wc_truth[2], t_wc_truth[0]))+90+yaw_wc_truth, np.sqrt(t_wc_truth[0]**2+t_wc_truth[2]**2))

# plt.scatter(alpha_centroid, d_centroid)
# plt.scatter(np.full(alpha_centroid.shape, np.degrees(np.arctan2(t_wc_truth[2], t_wc_truth[0]))+90+yaw_wc_truth), np.sqrt(t_wc_est[:, 0]**2+t_wc_est[:, 2]**2))
# plt.scatter(alpha_true_centroid, d_true_centroid)
# plt.scatter(np.degrees(np.arctan2(t_wc_truth[2], t_wc_truth[0]))+90+yaw_wc_truth, np.sqrt(t_wc_truth[0]**2+t_wc_truth[2]**2))

plt.scatter(t_wc_est[:, 0], t_wc_est[:, 2])
plt.scatter(t_wc_est_true_contour[:, 0], t_wc_est_true_contour[:, 2])

plt.show()

# print(cameraMatrix@T_cw_truth@target_3d)

# fig, ax = plt.subplots()
# ax.fill(target[:, 0], target[:, 1])
# ax.set_aspect('equal', 'box')
# ax.invert_yaxis()

# fig, ax = plt.subplots()
# ax.fill(target_c[0], target_c[1])
# ax.plot(panel_c[0], panel_c[1])
# ax.plot(np.append(target_c_n[0], target_c_n[0,0]), np.append(target_c_n[1], target_c_n[1,0]))
# ax.plot(np.append(reprojected_pts[0], reprojected_pts[0,0]), np.append(reprojected_pts[1], reprojected_pts[1,0]))

# ax.scatter(target_c_n_centroid[0], target_c_n_centroid[1])
# ax.set_xlim([0, 960])
# ax.set_ylim([0, 720])
# ax.invert_yaxis()
# ax.set_aspect('equal')
plt.show()