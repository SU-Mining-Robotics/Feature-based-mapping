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

def global_registration(source, target, voxel_size=0.05):
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    

    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, mutual_filter=True,
        max_correspondence_distance=voxel_size * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    return result.transformation

# Example Usage
if __name__ == "__main__":
    # Load two consecutive scans
    source_pcd = o3d.io.read_point_cloud("Start_pointcloud.ply")  # First scan
    target_pcd = o3d.io.read_point_cloud("second_scan.ply")  # Second scan

    # Remove ground plane
    source_pcd, _ = remove_ground_plane(source_pcd)
    target_pcd, _ = remove_ground_plane(target_pcd)
    
    print("Applying global registration")
    transformation = global_registration(source_pcd, target_pcd)
    print("Transformation is:")
    print(transformation)
    draw_registration_result(source_pcd, target_pcd, transformation)
