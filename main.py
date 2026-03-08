import argparse
import open3d as o3d
import numpy as np
import time
import registration
from registration import register, set_config_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    set_config_path(args.config)
    print(f"Using config file: {args.config}")

    # Load two misaligned point clouds from Open3D's demo dataset
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])  # source
    pcd.paint_uniform_color([1, 0.706, 0])

    pcd_transformed = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])  # target
    pcd_transformed.paint_uniform_color([0, 0.651, 0.929])

    # Apply extra transformation to increase misalignment
    additional_rotation = o3d.geometry.get_rotation_matrix_from_xyz([0.3, 0.5, 0.2])
    additional_translation = np.array([2.0, 1.5, 1.0])

    additional_transform = np.eye(4)
    additional_transform[:3, :3] = additional_rotation
    additional_transform[:3, 3] = additional_translation

    pcd.transform(additional_transform)

    print(f"Applied additional misalignment: rotation={[0.3, 0.5, 0.2]} rad, translation={additional_translation}")

    print("Visualizing source and target point clouds before registration.")
    o3d.visualization.draw_geometries(
        [pcd, pcd_transformed],
        window_name="Before Registration"
    )

    # Time the registration function
    start_time = time.time()
    transformation = register(pcd, pcd_transformed)
    end_time = time.time()

    print(f"Registration took {end_time - start_time:.4f} seconds.")

    # Apply estimated transformation
    pcd.transform(transformation)

    # Evaluate registration
    distance_threshold = registration.config["icp"]["distance_threshold"]
    evaluation = o3d.pipelines.registration.evaluate_registration(
        pcd, pcd_transformed, distance_threshold
    )

    print("Registration accuracy metrics:")
    print(f"  Fitness: {evaluation.fitness * 100:.2f} %")
    print(f"  Inlier RMSE: {evaluation.inlier_rmse:.4f}")
    print(f"  Correspondences found: {len(evaluation.correspondence_set)}")

    print("Visualizing source and target point clouds after registration.")
    o3d.visualization.draw_geometries(
        [pcd, pcd_transformed],
        window_name="After Registration"
    )


if __name__ == "__main__":
    main()
