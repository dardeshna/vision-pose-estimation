import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import scipy.optimize
import util
import time

rng = np.random.default_rng()

LL_params = {

    'camera_matrix' : np.array([
            [772.53876202, 0., 479.132337442],
            [0., 769.052151477, 359.143001808],
            [0., 0., 1.0]
        ]),
    'distortion_coeffs' : np.array(
            [2.9684613693070039e-01, -1.4380252254747885e+00, -2.2098421479494509e-03, -3.3894563533907176e-03, 2.5344430354806740e+00]
        ),
    'focal_length' : 2.9272781257541 #mm

}

class PoseEstimation():
    """Camera-based pose estimation simulation

    Input is a set of truth camera poses specified as a translation and rotation (yaw pitch roll) from the goal. Offsets
    can be added to create a set of adjusted camera poses (i.e. camera misalignment).

    Using the adjusted camera poses and a target model, image points are generated for the target. Gaussian pixel noise
    can be added to these image points.

    One of several different algorithms can then be called to estimate the camera pose using these noisy image points:

    1. "vanilla" solvePnP (cv.SOLVEPNP_ITERATIVE)
    2. 2D homography without refinement
    3. centroid -> distance + yaw -> constrained refinement
    4. solvePnP (cv.SOLVEPNP_IPPE) -> constrained refinement

    OpenCV coordinate system convention is used: +X is right, +Y is down and +Z is out of the camera. The class supports
    arbitrary shaped stacks of ndarrays (i.e. (M1,...,MM,<vector/matrix shape>)).
    """

    def __init__(self, camera_params=LL_params):
        """Initializes camera params and generates target
        """
        
        self.K = camera_params['camera_matrix']
        self.K_inv = np.linalg.inv(self.K)
        self.target_w, self.target_height = util.generate_target()

        self.results = {}

    def m_idx(self, x, idx, ndim=1):
        """Helper function to index into stack of ndarrays

        The stack x has shape (M1,...,MM,N1,...,NN) and an array of shape (N1,..,NN) is returned

        len(idx) + ndim = x.ndim

        Args:
            x (ndarray): stack of ndarrays
            idx (tuple): tuple of indices
            ndim (int, optional): number of dimensions of returned array

        Returns:
            ndarray: ndarray at idx
        """

        return np.broadcast_to(x, self.m+x.shape[-ndim:])[idx]

    def setup_truth(self, ypr_wc_truth, t_wc_truth, n=1):
        """Computes true pose of camera (T_wc) and generates noiseless image points

        Args:
            ypr_wc_truth (ndarray): yaw pitch roll of camera
            t_wc_truth (ndarray): translation of camera
            n (int, optional): number of repetitions over all points
        """

        self.m = np.broadcast_shapes(ypr_wc_truth.shape[:-1], t_wc_truth.shape[:-1])

        if n > 1:
            self.m = (n,)+self.m
        
        self.ypr_wc_truth = ypr_wc_truth
        self.t_wc_truth = t_wc_truth

        self.R_wc_truth = util.rotation_matrix_3d(self.ypr_wc_truth[...,0], self.ypr_wc_truth[...,1], self.ypr_wc_truth[...,2])
        
        self.T_wc_truth = util.Rt_to_T(self.R_wc_truth, self.t_wc_truth)
        self.T_cw_truth = np.linalg.inv(self.T_wc_truth)

        self.target_c = util.normalize_homogenous(self.K @ util.T_to_P(self.T_cw_truth) @ self.target_w)
        self.centroid_c = util.normalize_homogenous(self.K @ util.T_to_P(self.T_cw_truth) @ util.get_origin())

        self.ypr_c_offset = np.zeros((3,))
        self.t_c_offset = np.zeros((3,))
        
        self.target_c_noise = np.zeros(self.target_c.shape[-2:])
        self.calc_adj()

    def add_camera_offset(self, ypr_c_offset, t_c_offset):
        """Add an offset to the true camera pose (i.e. camera is misaligned)

        Args:
            ypr_c_offset (ndarray): yaw pitch roll to offset camera (note that angles are added to true ypr rather than composing two ypr rotations)
            t_c_offset (ndarray): translation to offset camera
        """

        self.ypr_c_offset = self.ypr_c_offset + ypr_c_offset
        self.t_c_offset = self.t_c_offset + t_c_offset
    
        self.calc_adj()

    def add_img_noise(self, sigma=1):
        """Add gaussian noise to image points

        Args:
            sigma (int, optional): standard deviation of noise to add (in pixels)
        """

        new_target_c_noise = np.zeros(self.m+self.target_c.shape[-2:])
        new_target_c_noise[...,:-1,:] = sigma * rng.standard_normal(size=new_target_c_noise[...,:-1,:].shape)

        self.target_c_noise = self.target_c_noise + new_target_c_noise

        self.calc_adj()

    def calc_adj(self):
        """Helper function to calculate adjusted pose of camera (T_wc) and generate noisy image points
        """

        self.ypr_wc_adj = self.ypr_wc_truth + self.ypr_c_offset
        self.t_wc_adj = self.t_wc_truth + self.t_c_offset

        self.R_wc_adj = util.rotation_matrix_3d(self.ypr_wc_adj[...,0], self.ypr_wc_adj[...,1], self.ypr_wc_adj[...,2])
        
        self.T_wc_adj = util.Rt_to_T(self.R_wc_adj, self.t_wc_adj)
        self.T_cw_adj = np.linalg.inv(self.T_wc_adj)

        self.target_c_adj = util.normalize_homogenous(self.K @ util.T_to_P(self.T_cw_adj) @ self.target_w)
        self.centroid_c_adj = util.normalize_homogenous(self.K @ util.T_to_P(self.T_cw_adj) @ util.get_origin())

        self.target_c_jit = self.target_c_noise + self.target_c_adj

    def get_empty_result(self):
        """Helper function to return an empty result dict
        """

        result = {
            'elapsed' : 0,
            'R_wc_est' : np.zeros(self.m+(3,3)),
            't_wc_est' : np.zeros(self.m+(3,)),
            'target_c_est' : np.zeros(self.m+self.target_c.shape[-2:]),
            'centroid_c_est' : np.zeros(self.m+self.centroid_c.shape[-2:])
        }

        return result

    def run_solve_pnp(self,name='solve_pnp'):
        """Estimates pose of camera using "vanilla" openCV solvePnP (cv.SOLVEPNP_ITERATIVE)
        
        1. compute a 2D homography H between (planar) world points and image points
        2. decompose H into R_cw and t_cw (rvec and tvec)
        3. iteratively refine R_cw and t_cw using Levenberg–Marquardt to minimize reprojection error
        """

        result = self.get_empty_result()
        start = time.time()

        for i in np.ndindex(self.m):

            target_c_jit = self.m_idx(self.target_c_jit, i, ndim=2)

            r_cw_est, t_cw_est = cv.solvePnP(util.from_homogenous(self.target_w).T, util.from_homogenous(target_c_jit).T, self.K, None)[1:]
            R_cw_est = cv.Rodrigues(r_cw_est)[0]
            P_cw_est = util.T_to_P(util.Rt_to_T(R_cw_est, np.squeeze(t_cw_est)))

            result['R_wc_est'][i] = R_cw_est.T
            result['t_wc_est'][i] = np.squeeze(-R_cw_est.T@t_cw_est)
            result['target_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ self.target_w)
            result['centroid_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ util.get_origin())

        result['elapsed'] = time.time()-start
        self.results[name] = result

    def run_homography(self,name='homography'):
        """Estimates pose of camera using 2D homography without refinement
        
        1. compute a 2D homography H between (planar) world points and image points
        2. decompose H into R_cw and t_cw
        """

        result = self.get_empty_result()
        start = time.time()

        for i in np.ndindex(self.m):

            target_c_jit = self.m_idx(self.target_c_jit, i, ndim=2)

            H = cv.findHomography(util.from_homogenous(self.target_w).T, util.from_homogenous(target_c_jit).T)[0]

            R_cw_est, t_cw_est = util.decompose_homography(H, self.K)
            P_cw_est = util.T_to_P(util.Rt_to_T(R_cw_est, t_cw_est))

            result['R_wc_est'][i] = R_cw_est.T
            result['t_wc_est'][i] = np.squeeze(-R_cw_est.T@t_cw_est)
            result['target_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ self.target_w)
            result['centroid_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ util.get_origin())

        result['elapsed'] = time.time()-start
        self.results[name] = result

    def run_centroid_solve(self,name='centroid_solve',use_homography=True):
        """Estimates pose of camera using distance + yaw to centroid and constrained refinement of skew
        
        use_homography=True:
        1. compute a 2D homography H between (planar) world points and image points
        2. use homography to determine centroid of target in image
        use_homography=False:
        1. use contour moments to estimate centroid of target

        pose calculation:
        1. determine distance (d) and yaw (alpha) to centroid using atan + pythagorean theorem + known heights + camera pitch
        2. assume camera is in front of target (no skew)
        3. refine skew using Levenberg–Marquardt to minimize reprojection error
        """

        result = self.get_empty_result()
        start = time.time()

        for i in np.ndindex(self.m):

            target_c_jit = self.m_idx(self.target_c_jit, i, ndim=2)

            if use_homography:
                H = cv.findHomography(util.from_homogenous(self.target_w).T, util.from_homogenous(target_c_jit).T)[0]
                centroid_c_est = H[0:2, 2]
            else:
                moments = cv.moments(util.from_homogenous(target_c_jit).T.astype(np.float32))
                centroid_c_est = np.array([moments['m10'], moments['m01']])/moments['m00']

            pitch_wc_truth = self.m_idx(self.ypr_wc_truth, i)[1]
            y_wc_truth = self.m_idx(self.t_wc_truth, i)[1]
            
            intermediate = util.rotation_matrix_3d(pitch=pitch_wc_truth) @ self.K_inv @ np.append(centroid_c_est, 1)
            alpha = np.degrees(np.arctan2(-intermediate[0], intermediate[2]))
            d = -(y_wc_truth)/intermediate[1]*np.sqrt(intermediate[0]**2+intermediate[2]**2)

            R_wc_prime_est = util.rotation_matrix_3d(yaw=alpha, pitch=pitch_wc_truth)
            T_cw_prime_est = util.Rt_to_T(R_wc_prime_est.T, -R_wc_prime_est.T@np.array([0, y_wc_truth, -d]))

            T_cw_est = lambda skew : T_cw_prime_est @ util.Rt_to_T(util.rotation_matrix_3d(yaw=-skew), np.zeros((3,)))

            f = lambda skew: (util.from_homogenous(target_c_jit) - util.from_homogenous(self.K @ util.T_to_P(T_cw_est(skew)) @ self.target_w)).ravel()
            skew = scipy.optimize.least_squares(f, 0, method='lm')['x']
            
            result['R_wc_est'][i], result['t_wc_est'][i] = util.T_to_Rt(np.linalg.inv(T_cw_est(skew)))

            P_cw_est = util.T_to_P(T_cw_est(skew))

            result['target_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ self.target_w)
            result['centroid_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ util.get_origin())

        result['elapsed'] = time.time()-start
        self.results[name] = result

    def run_constrained_lm(self,name='constrained_lm'):
        """Estimates pose of camera using openCV solvePnP (cv.SOLVEPNP_IPPE) and constrained refinement
        
        1. use IPPE method to obtain initial R_cw and t_cw (https://link.springer.com/article/10.1007%2Fs11263-014-0725-5)
        2. given known heights + camera pitch, refine x, z, yaw using Levenberg–Marquardt to minimize reprojection error
        """

        result = self.get_empty_result()
        start = time.time()

        for i in np.ndindex(self.m):

            target_c_jit = self.m_idx(self.target_c_jit, i, ndim=2)

            r_cw_est, t_cw_est = cv.solvePnP(util.from_homogenous(self.target_w).T, util.from_homogenous(target_c_jit).T, self.K, None, flags=cv.SOLVEPNP_IPPE)[1:]
            R_cw_est = cv.Rodrigues(r_cw_est)[0]
            P_cw_est = util.T_to_P(util.Rt_to_T(R_cw_est, np.squeeze(t_cw_est)))

            R_wc_est = R_cw_est.T
            t_wc_est = -R_wc_est@np.squeeze(t_cw_est)
            yaw_wc_est = np.degrees(np.arctan2(R_wc_est[0,2], R_wc_est[2,2]))
            
            pitch_wc_truth = self.m_idx(self.ypr_wc_truth, i)[1]
            roll_wc_truth = self.m_idx(self.ypr_wc_truth, i)[2]
            y_wc_truth = self.m_idx(self.t_wc_truth, i)[1]

            T_cw_est = lambda yaw, x, z : np.linalg.inv(util.Rt_to_T(util.rotation_matrix_3d(yaw=yaw, pitch=pitch_wc_truth, roll=roll_wc_truth), np.array([x, y_wc_truth, z])))

            f = lambda x: (util.from_homogenous(target_c_jit) - util.from_homogenous(self.K @ util.T_to_P(T_cw_est(x[0], x[1], x[2])) @ self.target_w)).ravel()
            res = scipy.optimize.least_squares(f, [yaw_wc_est, t_wc_est[0], t_wc_est[2]], method='lm')['x']
            
            result['R_wc_est'][i], result['t_wc_est'][i] = util.T_to_Rt(np.linalg.inv(T_cw_est(res[0], res[1], res[2])))

            P_cw_est = util.T_to_P(T_cw_est(res[0], res[1], res[2]))

            result['target_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ self.target_w)
            result['centroid_c_est'][i] = util.normalize_homogenous(self.K @ P_cw_est @ util.get_origin())

        result['elapsed'] = time.time()-start
        self.results[name] = result

    def plot_goal(self,runs=None):
        """Goal visualization for first camera pose in stack

        Args:
            runs (list, optional): list of run names
        """

        idx = tuple([0,] * len(self.m))

        T_cw_adj = util.T_to_P(self.m_idx(self.T_cw_adj, idx, ndim=2))

        panel_w = np.vstack((np.array([[-20, -20, 20, 20, -20], [self.target_height, -30, -30, self.target_height, self.target_height]]), np.zeros(5), np.ones(5))) 
        panel_c_adj = util.from_homogenous(self.K @ T_cw_adj @ panel_w)

        target_c_adj = util.from_homogenous(self.m_idx(self.target_c_adj, idx, ndim=2))
        centroid_c_adj = util.from_homogenous(self.m_idx(self.centroid_c_adj, idx, ndim=2))

        target_c_jit = util.from_homogenous(self.m_idx(self.target_c_jit, idx, ndim=2))
        
        fig, ax = plt.subplots()

        ax.fill(target_c_adj[0], target_c_adj[1])
        ax.plot(panel_c_adj[0], panel_c_adj[1])

        ax.plot(np.append(target_c_jit[0], target_c_jit[0,0]), np.append(target_c_jit[1], target_c_jit[1,0]), label='detected target')
        ax.scatter(centroid_c_adj[0], centroid_c_adj[1], label='centroid')

        for k in (runs if runs else self.results.keys()):

            target_c_est = util.from_homogenous(self.results[k]['target_c_est'][idx])
            centroid_c_est = util.from_homogenous(self.results[k]['centroid_c_est'][idx])

            ax.plot(np.append(target_c_est[0], target_c_est[0,0]), np.append(target_c_est[1], target_c_est[1,0]), '--', label=k)
            ax.scatter(centroid_c_est[0], centroid_c_est[1], label=k)
        
        ax.set_xlim([0, 960])
        ax.set_ylim([0, 720])
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('view from camera')

        return fig, ax

    def plot_dispersion(self,runs=None):
        """Top down view of estimated pose (x and z)

        Args:
            runs (list, optional): list of run names
        """

        fig, ax = plt.subplots()

        for k in (runs if runs else self.results.keys()):
            t_wc_est = self.results[k]['t_wc_est']
            ax.scatter(t_wc_est[...,0].ravel(), t_wc_est[...,2].ravel(), label=k)

        ax.scatter(t_wc_truth[...,0].ravel(), t_wc_truth[...,2].ravel(), label='truth')

        ax.legend()
        ax.set_title('field position')

        return fig, ax

    def plot_violin(self,runs=None,filter=False):
        """Violin plot of distributions for rotation and translation components of estimated pose

        Args:
            runs (list, optional): list of run names
            filter (bool, optional): whether to include estimates with negative y (camera higher than target)
        """

        fig_t, ax_t = plt.subplots(3,1)
        fig_r, ax_r = plt.subplots(3,1)

        for j, k in enumerate(runs if runs else self.results.keys()):
            t_wc_est = self.results[k]['t_wc_est']
            ypr_wc_est = util.R_to_ypr(self.results[k]['R_wc_est'])

            if filter:
                f = t_wc_est[...,1] > 0
            else:
                f = np.ones(t_wc_est.shape[:-1], dtype=np.bool8)

            for i in range(t_wc_est.shape[-1]):
                ax_t[i].violinplot(t_wc_est[f,i], positions=[j,])

            for i in range(ypr_wc_est.shape[-1]):
                ax_r[i].violinplot(ypr_wc_est[f,i], positions=[j,])

        # ax_t[0].legend()
        ax_t[0].set_title('translation')
        for i, l in enumerate(('x', 'y', 'z')):
            ax_t[i].set_ylabel(l)
        
        # ax_r[0].legend()
        ax_r[0].set_title('rotation')
        for i, l in enumerate(('yaw', 'pitch', 'roll')):
            ax_r[i].set_ylabel(l)

        return (fig_t, fig_r), (ax_t, ax_r)

    def print_elapsed(self,runs=None):
        """Computation time for individual algorithms
        """

        for k in (runs if runs else self.results.keys()):
            print(f"{k}: {self.results[k]['elapsed']} seconds")


if __name__ == "__main__":

    pose_estimation = PoseEstimation()

    yaw_wc_truth = 10
    pitch_wc_truth = 35
    x_wc_truth = -3*12
    z_wc_truth = -10*12
    camera_height = 32

    n = 100

    ypr_wc_truth = np.array([yaw_wc_truth, pitch_wc_truth, 0])
    # ypr_wc_truth = np.array([[yaw_wc_truth, pitch_wc_truth, 0], [yaw_wc_truth*2, pitch_wc_truth, 0]])
    t_wc_truth = np.array([x_wc_truth, pose_estimation.target_height-camera_height, z_wc_truth])
    # t_wc_truth = np.array([[x_wc_truth, pose_estimation.target_height-camera_height, z_wc_truth], [x_wc_truth+12, pose_estimation.target_height-camera_height, z_wc_truth]])

    pose_estimation.setup_truth(ypr_wc_truth, t_wc_truth, n)
    # pose_estimation.add_camera_offset(np.array([0, 1, 0]), np.array([0, 0, 0]))
    pose_estimation.add_img_noise()

    # print(pose_estimation.T_wc_adj-pose_estimation.T_wc_truth)

    pose_estimation.run_solve_pnp()
    # pose_estimation.run_homography()
    pose_estimation.run_centroid_solve()
    pose_estimation.run_constrained_lm()
    pose_estimation.plot_goal()
    pose_estimation.plot_dispersion()
    # pose_estimation.plot_violin()
    pose_estimation.print_elapsed()

    plt.show()