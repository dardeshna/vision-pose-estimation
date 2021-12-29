import numpy as np
import cv2 as cv

def rotation_matrix_2d(theta=0):

    theta = np.radians(theta)
    
    theta_cos = np.cos(theta)
    theta_sin = np.sin(theta)

    R = np.zeros(theta.shape + (2,2))

    R[...,0,0] = theta_cos
    R[...,0,1] = -theta_sin
    R[...,1,0] = theta_sin
    R[...,1,1] = theta_cos

    return R

def rotation_matrix_3d(yaw=0, pitch=0, roll=0):
    
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    yaw_cos = np.cos(yaw)
    yaw_sin = np.sin(yaw)
    pitch_cos = np.cos(pitch)
    pitch_sin = np.sin(pitch)
    roll_cos = np.cos(roll)
    roll_sin = np.sin(roll)

    # cos 0 sin
    # 0 1 0
    # -sin 0 cos
    
    Ry = np.zeros(yaw.shape + (3,3))
    Ry[...,0,0] = yaw_cos
    Ry[...,0,2] = yaw_sin
    Ry[...,1,1] = 1
    Ry[...,2,0] = -yaw_sin
    Ry[...,2,2] = yaw_cos

    # 1 0 0
    # 0 cos -sin
    # 0 sin cos

    Rx = np.zeros(pitch.shape + (3,3))
    Rx[...,0,0] = 1
    Rx[...,1,1] = pitch_cos
    Rx[...,1,2] = -pitch_sin
    Rx[...,2,1] = pitch_sin
    Rx[...,2,2] = pitch_cos

    # cos -sin 0
    # sin cos 0
    # 0 0 1

    Rz = np.zeros(roll.shape + (3,3))
    Rz[...,0,0] = roll_cos
    Rz[...,0,1] = -roll_sin
    Rz[...,1,0] = roll_sin
    Rz[...,1,1] = roll_cos
    Rz[...,2,2] = 1

    return Ry@Rx@Rz

def R_to_ypr(R):

    yaw = np.degrees(np.arctan2(R[...,0,2], R[...,2,2]))
    pitch = np.degrees(np.arctan2(-R[...,1,2], np.sqrt(R[...,1,0]**2 + R[...,1,1]**2)))
    roll = np.degrees(np.arctan2(R[...,1,0], R[...,1,1]))

    return np.concatenate((yaw[...,None], pitch[...,None], roll[...,None]), axis=-1)

def Rt_to_T(R, t):

    sz = np.broadcast_shapes(R.shape[:-2], t.shape[:-1])
    R = np.broadcast_to(R, sz+R.shape[-2:])
    t = np.broadcast_to(t, sz+t.shape[-1:])

    return np.concatenate((np.concatenate((R, t[...,None]), -1), np.broadcast_to(np.array([0,]*R.shape[-1]+[1,]), R.shape[:-2]+(1,R.shape[-1]+1))),-2)

def T_to_Rt(T):

    return T[...,:-1,:-1], T[...,:-1,-1]

def T_to_P(T):

    return T[...,:-1,:]

def to_homogenous(x):

    return np.concatenate((x, np.broadcast_to(1, x.shape[:-2]+(1,x.shape[-1]))),0)

def from_homogenous(x):

    return x[...,:-1,:]/x[...,-1,None,:]

def normalize_homogenous(x):

    return x[...,:,:]/x[...,-1,None,:]

def decompose_homography(H, K):

    H = np.linalg.solve(K, H)

    t = 2 * H[:,2]/(np.linalg.norm(H[:,0])+np.linalg.norm(H[:,1]))

    H[:,0:2] = H[:,0:2]/np.linalg.norm(H[:,0:2], axis=0)
    R = np.hstack((H[:,0:2], np.atleast_2d(np.cross(H[:,0],H[:,1])).T))

    U, S, Vh = np.linalg.svd(R)
    R = U@Vh

    return R, t

def generate_target():

    target = np.zeros((8, 2))
    target[0] = np.array([17, 0])
    target[1] = rotation_matrix_2d(-60)@target[0]
    target[2:4] = target[1::-1]*np.array([-1, 1])
    target[4:8] = target[3::-1]*(target[1, 1]+2)/target[1, 1]
    target = target*np.array([1, -1]) # flip y axis
    target_moments = cv.moments(target.astype(np.float32))
    target_centroid_y = target_moments['m01']/target_moments['m00']
    target_height = 6*12+9.25+target[1, 1]-target_centroid_y
    target_3d = np.vstack((target[:, 0], target[:, 1]-target_centroid_y, np.zeros(8), np.ones(8)))

    return target_3d, target_height

def get_origin(ndim=3):

    return np.array([0,]*ndim+[1,])[...,None]

if __name__ == "__main__":

    res = Rt_to_T(np.repeat(rotation_matrix_2d(10)[None,...], 3, 0), np.repeat(np.array([1,2])[None,...], 3, 0))

    print(res)
    print(T_to_Rt(res))

    print(rotation_matrix_2d(theta=[10,20,30]))
    print(rotation_matrix_3d(yaw=[10,20,30]))
    print(rotation_matrix_3d(yaw=[10,20,30], pitch=[[0,20,-20],[10,20,30]], roll=[[0],[10]]))