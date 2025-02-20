import cv2
import matplotlib.pyplot as plt

def orb_feature_matching(img1_path, img2_path):
    # Load the images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    print(f"Keypoints in Image 1: {len(keypoints1)}")
    print(f"Keypoints in Image 2: {len(keypoints2)}")
   
    #Print feature locations
    for i in range(len(keypoints1)):
        print(f"Keypoint {i} in Image 1: {keypoints1[i].pt}")
        
    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches
    result_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(result_img)
    plt.title("Feature Matching with ORB")
    plt.axis('off')
    plt.show()

# Example usage
# Provide paths to two images for matching
img1_path = "src/voxel_slam/Tests/Image_based/Image1.jpg"  # Replace with your image path
img2_path = "src/voxel_slam/Tests/Image_based/Image2.jpg"  # Replace with your image path

orb_feature_matching(img1_path, img2_path)
