import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# FLANN parameters for matching
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Variables to store previous frame keypoints and descriptors
prev_kp, prev_des = None, None

try:
    while True:
        # Get frameset from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp, des = sift.detectAndCompute(gray, None)

        if prev_kp is not None and prev_des is not None and des is not None:
            # Match descriptors using FLANN matcher
            matches = flann.knnMatch(prev_des, des, k=2)

            # Apply ratio test as per Lowe's paper
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Draw matches
            match_frame = cv2.drawMatches(prev_frame, prev_kp, frame, kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Show matched keypoints
            cv2.imshow("Feature Matching", match_frame)

        # Update previous frame keypoints and descriptors
        prev_kp, prev_des, prev_frame = kp, des, frame.copy()

        # Exit loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
