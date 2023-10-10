import cv2
import numpy as np


class Camera(object):
    def __init__(self):

        self.intrinsic = Camera.Intrinsic()
        self.extrinsic = Camera.Extrinsic()

    @property
    def P(self):
        E = np.zeros((3, 4))
        E[:3, :3] = self.extrinsic.R
        E[:3, 3] = self.extrinsic.T
        return self.intrinsic.A @ E

    ################# Base init settings ######################
    @staticmethod
    def compute_A(f, c):
        """
        :return: Intrinsic matrix from focal length and principal point
        :rtype: np.ndarray
        """
        return np.array(
            [
                [f[0], 0, c[0]],
                [0, f[1], c[1]],
                [0, 0, 1],
            ]
        )

    class Extrinsic(object):
        def __init__(self):
            # Translation / rvecs
            self._T = np.array([0, 0, 0], dtype=float)

            # Rotation matrix
            self._R = np.eye(3)

            # Tpr / rvecs in euler angles in radians [tilt, pan, roll]
            self._tpr = np.array([0, 0, 0], dtype=float)

        @property
        def T(self):
            """:rtype: np.ndarray"""
            return self._T

        @T.setter
        def T(self, v):
            self._T = np.array(v)

        @property
        def R(self):
            """:rtype: np.ndarray"""
            if self._R is None:
                if self._tpr is not None:
                    self._R = cv2.Rodrigues(self._tpr)[0]
            return self._R

        @R.setter
        def R(self, v):
            self._tpr = None
            self._R = np.array(v)

        @property
        def tpr(self):
            """:rtype: np.ndarray"""
            if self._tpr is None:
                if self.R is not None:
                    self._tpr = cv2.Rodrigues(self._R)[0][:, 0]
            return self._tpr

        @tpr.setter
        def tpr(self, v):
            self._tpr = np.array(v)
            self._R = None

    class Intrinsic(object):
        def __init__(self):
            # Focal lengths in pixels (f_u, f_v)
            self._f = np.array([1, 1], dtype=float)

            # Principal point in pixels (c_u, c_v)
            self._c = np.array([0, 0], dtype=float)

            # Radial distortion coefficients (k_1, k_2)
            self._k = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

            # Intrinsic transformation matrix (OpenCV version)
            self._A = np.zeros((3, 3), dtype=float)

        @property
        def f(self):
            """:rtype: np.ndarray"""
            return self._f

        @f.setter
        def f(self, v):
            self._f = np.asarray(v, dtype=float)
            self._A = None
            self._A_inv = None

        @property
        def c(self):
            """:rtype: np.ndarray"""
            return self._c

        @c.setter
        def c(self, v):
            self._c = np.asarray(v, dtype=float)
            self._A = None
            self._A_inv = None

        @property
        def k(self):
            """:rtype: np.ndarray"""
            return self._k

        @k.setter
        def k(self, v):
            self._k = np.asarray(v, dtype=float)

        @property
        def A(self):
            """:rtype: np.ndarray"""
            if self._A is None:
                self._A = Camera.compute_A(self.f, self.c)
            return self._A

    ################ Base projection functions ###################

    def project(self, points):
        """
        Projects 3D points into image plane of camera
        :param points: n x 3 array of points
        :return: n x 2 array of points
        """
        # Make sure its a numpy array
        points = np.asarray(points, dtype=np.float32)

        # Handle sequences
        if len(points.shape) == 3:
            p_shape = points.shape
            img_pts, jacobian = cv2.projectPoints(
                objectPoints=points.reshape((-1, 3)),
                rvec=self.extrinsic.tpr,
                tvec=self.extrinsic.T,
                cameraMatrix=self.intrinsic.A,
                distCoeffs=self.intrinsic.k,
            )
            img_pts = img_pts.reshape((p_shape[0], p_shape[1], 2))
            return img_pts

        img_pts, jacobian = cv2.projectPoints(
            objectPoints=points,
            rvec=self.extrinsic.tpr,
            tvec=self.extrinsic.T,
            cameraMatrix=self.intrinsic.A,
            distCoeffs=self.intrinsic.k,
        )

        return img_pts.squeeze()
    
    def camera_to_dict(self):
        # Convert camera to dict such that it can be used in a pytorch dataloader
        camera_dict = {
            "intrinsic": {
                "f": self.intrinsic.f,
                "c": self.intrinsic.c,
                "k": self.intrinsic.k,
            },
            "extrinsic": {
                "T": self.extrinsic.T,
                "R": self.extrinsic.R,
                "tpr": self.extrinsic.tpr,
            },
        }
        return camera_dict
