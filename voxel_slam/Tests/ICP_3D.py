import open3d as o3d
import numpy as np
import copy

def remove_ground_plane(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, 
                                             ransac_n=ransac_n, 
                                             num_iterations=num_iterations)
    ground = pcd.select_by_index(inliers)
    filtered_pcd = pcd.select_by_index(inliers, invert=True)
    return filtered_pcd, ground

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# Example Usage
if __name__ == "__main__":
    # Load two consecutive scans
    source_pcd = o3d.io.read_point_cloud("Start_pointcloud.ply")  # First scan
    target_pcd = o3d.io.read_point_cloud("second_scan.ply")  # Second scan

    # Remove ground plane
    source_pcd, _ = remove_ground_plane(source_pcd)
    target_pcd, _ = remove_ground_plane(target_pcd)
    
    threshold = 0.02  # Stopping criteria threshold
    trans_init = np.identity(4)
    x_translation = -0.1
    y_translation = 2.0
    z_translation = 0.0
    pitch = 0.0
    roll = 0.0
    yaw = 0.1
    
    trans_init[0, 3] = x_translation
    trans_init[1, 3] = y_translation
    trans_init[2, 3] = z_translation
    trans_init[0, 0] = np.cos(yaw) 
    trans_init[0, 1] = -np.sin(yaw) 
    trans_init[1, 0] =  np.sin(yaw)
    trans_init[1, 1] = np.cos(yaw)
    # trans_init[0, 1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    # trans_init[0, 2] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    
    print("No alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source_pcd, target_pcd, threshold, np.identity(4))
    print(evaluation)
    draw_registration_result(source_pcd, target_pcd, np.identity(4))
    
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source_pcd, target_pcd, threshold, trans_init)
    print(evaluation)
    draw_registration_result(source_pcd, target_pcd, trans_init)
    
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source_pcd, target_pcd, reg_p2p.transformation)
    
    print("Apply point-to-point ICP with max iterations")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source_pcd, target_pcd, reg_p2p.transformation)
    
    print("Apply point-to-plane ICP")
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source_pcd, target_pcd, reg_p2l.transformation)
