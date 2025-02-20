import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

# Start streaming
pipeline.start(config)

# Get camera intrinsic parameters
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Camera intrinsic matrix for solvePnP
K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
              [0, intrinsics.fy, intrinsics.ppy],
              [0, 0, 1]])

# Initialize SIFT detector
sift = cv2.SIFT_create()

# FLANN matcher parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Variables to store previous frame keypoints and descriptors
prev_kp, prev_des, prev_pts3D = None, None, None

try:
    while True:
        # Get frameset from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        frame = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = sift.detectAndCompute(gray, None)

        # Convert keypoints to numpy array
        keypoints_np = np.array([kp.pt for kp in kp], dtype=np.float32)

        # Extract depth for keypoints and convert to 3D points
        points_3D = []
        for i, (x, y) in enumerate(keypoints_np):
            depth_value = depth[int(y), int(x)] * 0.001  # Convert depth to meters
            if depth_value > 0:
                X = (x - intrinsics.ppx) * depth_value / intrinsics.fx
                Y = (y - intrinsics.ppy) * depth_value / intrinsics.fy
                Z = depth_value
                points_3D.append((X, Y, Z))
            else:
                points_3D.append(None)  # Ignore points with no depth

        points_3D = np.array([p for p in points_3D if p is not None], dtype=np.float32)

        # Debug: Print number of points
        print(f"3D Points: {len(points_3D)}")

        if prev_kp is not None and prev_des is not None and des is not None and len(points_3D) > 0:
            # Match descriptors using FLANN matcher
            matches = flann.knnMatch(prev_des, des, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            prev_matched_pts = []
            curr_matched_pts = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    prev_matched_pts.append(prev_kp[m.queryIdx].pt)
                    curr_matched_pts.append(kp[m.trainIdx].pt)

            prev_matched_pts = np.array(prev_matched_pts, dtype=np.float32)
            curr_matched_pts = np.array(curr_matched_pts, dtype=np.float32)

            # Debug: Print the number of matches
            print(f"Good Matches: {len(good_matches)}")
            print(f"current matched points: {len(curr_matched_pts)}")

            if len(prev_matched_pts) >= 4 and len(prev_pts3D) >= 4:
                # Convert to numpy arrays and ensure they have the same shape
                prev_pts3D = np.array(prev_pts3D, dtype=np.float32)
                curr_matched_pts = np.array(curr_matched_pts, dtype=np.float32)

                print(f"prev_pts3D shape: {prev_pts3D.shape}, dtype: {prev_pts3D.dtype}")
                print(f"curr_matched_pts shape: {curr_matched_pts.shape}, dtype: {curr_matched_pts.dtype}")

                # Ensure the number of points match
                if prev_pts3D.shape[0] == curr_matched_pts.shape[0]:
                    _, rvec, tvec, _ = cv2.solvePnPRansac(prev_pts3D, curr_matched_pts, K, None)

                    # Convert rotation vector to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)

                    print(f"Rotation Matrix:\n{R}")
                    print(f"Translation Vector:\n{tvec.T}")
                else:
                    print("Mismatch between 3D points and 2D matched points!")
            else:
                print("Not enough valid points for solvePnPRansac")

            # Draw matches
            match_frame = cv2.drawMatches(prev_frame, prev_kp, frame, kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Show matched keypoints
            cv2.imshow("Feature Matching with Depth", match_frame)

        # Update previous frame data
        prev_kp, prev_des, prev_pts3D, prev_frame = kp, des, points_3D, frame.copy()

        # Exit loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
