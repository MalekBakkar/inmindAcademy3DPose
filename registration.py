import open3d as o3d
import numpy as np
import yaml


CONFIG_PATH = "config/config.yaml"
config = None


def load_config(path=None):
    global config, CONFIG_PATH
    if path is not None:
        CONFIG_PATH = path
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_config_path(path):
    load_config(path)


# Load default config when module is imported
load_config()


# -----------------------------
# Preprocess point cloud
# -----------------------------
def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * config["normals"]["radius_multiplier"]
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=config["normals"]["max_nn"]
        )
    )

    radius_feature = voxel_size * config["features"]["radius_multiplier"]
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=config["features"]["max_nn"]
        )
    )

    return pcd_down, pcd_fpfh


# -----------------------------
# Global Registration (RANSAC)
# -----------------------------
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):

    distance_threshold = voxel_size * config["ransac"]["distance_multiplier"]

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        config["ransac"]["ransac_n"],
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                config["ransac"]["edge_length_threshold"]
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            )
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            config["ransac"]["max_iterations"],
            config["ransac"]["confidence"]
        )
    )

    return result


# -----------------------------
# Registration Pipeline
# -----------------------------
def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Align pcd1 (source) to pcd2 (target)

    Steps:
    1) Global registration (FPFH + RANSAC)
    2) ICP refinement with robust kernel
    """

    source = pcd1
    target = pcd2

    voxel_size = config["voxel_size"]

    # -----------------------------
    # Global registration
    # -----------------------------
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = execute_global_registration(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        voxel_size
    )

    trans_init = result_ransac.transformation

    # -----------------------------
    # Estimate normals for ICP
    # -----------------------------
    radius_normal = voxel_size * config["normals"]["radius_multiplier"]

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=config["normals"]["max_nn"]
        )
    )

    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=config["normals"]["max_nn"]
        )
    )

    # -----------------------------
    # Robust ICP refinement
    # -----------------------------
    threshold = config["icp"]["distance_threshold"]

    loss = o3d.pipelines.registration.TukeyLoss(
        k=config["robust_kernel"]["k"]
    )

    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        p2l,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config["icp"]["max_iterations"]
        )
    )

    return reg_p2l.transformation
