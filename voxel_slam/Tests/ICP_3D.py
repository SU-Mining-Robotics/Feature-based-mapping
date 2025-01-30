import open3d as o3d
import numpy as np
import copy


class ICP:
    def __init__(self, source_pcd, target_pcd, voxel_size=0.05, max_iterations=50):
        self.source_pcd = source_pcd
        self.target_pcd = target_pcd
        self.voxel_size = voxel_size
        self.max_iterations = max_iterations
    
    @staticmethod
    def apply_icp(self, source_pcd, target_pcd, voxel_size=0.05, max_iterations=50):
        """
        Apply ICP to align the source point cloud to the target.

        Parameters:
            source_pcd (o3d.geometry.PointCloud): Source point cloud.
            target_pcd (o3d.geometry.PointCloud): Target point cloud.
            voxel_size (float): Voxel size for downsampling.
            max_iterations (int): Maximum number of ICP iterations.

        Returns:
            o3d.geometry.PointCloud: Transformed source point cloud.
            np.ndarray: Transformation matrix.
        """
        # Downsample the point clouds
        source_down = self.source_pcd.voxel_down_sample(self.voxel_size)
        target_down = self.target_pcd.voxel_down_sample(self.voxel_size)

        # Estimate normals for better alignment
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))

        # Apply ICP
        print("Applying ICP...")
        icp_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_correspondence_distance=self.voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
        )
        print("ICP converged:", icp_result.fitness > 0.8)
        print("Transformation matrix:\n", icp_result.transformation)

        # Transform the source point cloud
        source_transformed = self.source_pcd.transform(icp_result.transformation)
        return source_transformed, icp_result.transformation
    
def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])


# Example Usage
if __name__ == "__main__":
    # Load two consecutive scans
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source_pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])  # Replace with your first scan
    target_pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])  # Replace with your second scan
    threshold = 0.02                                              # Threshold for stopping criteria
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],         # Initial transformation matrix (From something like odometry)
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], 
                            [0.0, 0.0, 0.0, 1.0]])
    
    #Initliaze ICP class
    icp = ICP(source_pcd, target_pcd)
    
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source_pcd, target_pcd, threshold, trans_init)
    print(evaluation)
    
    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
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
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source_pcd, target_pcd, reg_p2l.transformation)

    # Align the scans
    transformed_pcd, transformation = icp.apply_icp(source_pcd, target_pcd, voxel_size=0.1)
    draw_registration_result(transformed_pcd, target_pcd, transformation)
   
