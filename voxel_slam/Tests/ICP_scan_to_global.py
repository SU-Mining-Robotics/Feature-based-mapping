import numpy as np
import matplotlib.pyplot as plt
from Utils import wrapAngle

# Edges: Global map (N,2), N points of [x, y]
# Scan: Local scan  (M,2), M points of [x, y]
# Pose: Initial pose guess (x, y, orientation)

class ICPScanMatcher:
    def __init__(self, eps=0.0001, max_iter=100, ransac_trials=15, show_animation=True):
        self.eps = eps
        self.max_iter = max_iter
        self.ransac_trials = ransac_trials
        self.show_animation = show_animation

        if self.show_animation:
            self.fig = plt.figure()

    def match(self, edges, scan, pose):
        if len(scan) < 5 or len(edges) < len(scan):
            return None

        scan = np.unique(scan, axis=0)  # Remove duplicate points

        edges = edges.T
        scan = scan.T

        H = np.eye(3)  # Homogeneous transformation matrix

        dError = np.inf
        preError = np.inf
        count = 0

        while dError >= self.eps:
            count += 1

            indexes, total_error = self._nearest_neighbor(edges, scan)
            edges_matched = edges[:, indexes]

            if self.show_animation:
                self._plot_points(edges_matched, scan)

            best_Rt, best_Tt, min_error = self._ransac(edges_matched, scan)

            scan = (best_Rt @ scan) + best_Tt[:, np.newaxis]

            dError = preError - total_error
            preError = total_error

            H = self._update_homogeneous_matrix(H, best_Rt, best_Tt)

            if count >= self.max_iter:
                break

        R = H[0:2, 0:2]
        T = H[0:2, 2]

        if abs(T[0]) > 5 or abs(T[1]) > 5:
            return None
        else:
            x = pose[0] + T[0]
            y = pose[1] + T[1]
            orientation = wrapAngle(pose[2] + np.arctan2(R[1, 0], R[0, 0]))

            return np.array([x, y, orientation])

    def _update_homogeneous_matrix(self, Hin, R, T):
        H = np.eye(3)
        H[0:2, 0:2] = R
        H[0:2, 2] = T
        return Hin @ H

    def _nearest_neighbor(self, edges, scan):
        d = np.linalg.norm(np.repeat(scan, edges.shape[1], axis=1) - np.tile(edges, (1, scan.shape[1])), axis=0)
        d = d.reshape(scan.shape[1], edges.shape[1])

        indexes = np.argmin(d, axis=1)
        error = np.min(d, axis=1)

        return indexes, np.sum(error)

    def _ransac(self, edges_matched, scan):
        min_error = float('inf')
        best_Rt, best_Tt = None, None

        for _ in range(self.ransac_trials):
            sample = np.random.choice(scan.shape[1], 5, replace=False)
            Rt, Tt = self._svd_motion_estimation(edges_matched[:, sample], scan[:, sample])
            temp_points = (Rt @ scan) + Tt[:, np.newaxis]
            _, error = self._nearest_neighbor(edges_matched, temp_points)

            if error < min_error:
                min_error = error
                best_Rt, best_Tt = Rt, Tt

        return best_Rt, best_Tt, min_error

    def _svd_motion_estimation(self, prev_points, curr_points):
        pm = np.mean(prev_points, axis=1)
        cm = np.mean(curr_points, axis=1)

        p_shift = prev_points - pm[:, np.newaxis]
        c_shift = curr_points - cm[:, np.newaxis]

        W = c_shift @ p_shift.T
        u, _, vh = np.linalg.svd(W)

        R = (u @ vh).T
        T = pm - (R @ cm)

        return R, T

    def _plot_points(self, edges, scan):
        plt.cla()
        plt.plot(edges[0, :], edges[1, :], ".r", markersize=1, label="Global Map")
        plt.plot(scan[0, :], scan[1, :], ".b", markersize=1, label="Local Scan")
        plt.plot(0.0, 0.0, "xr", label="Origin")
        plt.axis("equal")
        plt.legend()
        plt.pause(0.01)

# Example usage
def main():
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    radius = 10
    edges = np.vstack((radius * np.cos(theta), radius * np.sin(theta)))

    local_scan_size = 30
    scan_indices = np.random.choice(num_points, local_scan_size, replace=False)
    scan = edges[:, scan_indices]

    angle = np.deg2rad(30)
    translation = np.array([2.0, -3.0])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    transformed_scan = (rotation_matrix @ scan) + translation[:, np.newaxis]

    initial_pose = np.array([0.0, 0.0, 0.0])

    matcher = ICPScanMatcher(show_animation=True)
    estimated_pose = matcher.match(edges.T, transformed_scan.T, initial_pose)

    if estimated_pose is not None:
        print(f"Estimated Pose: x={estimated_pose[0]:.2f}, y={estimated_pose[1]:.2f}, orientation={np.rad2deg(estimated_pose[2]):.2f}°")
    else:
        print("ICP failed to converge.")

if __name__ == "__main__":
    main()
