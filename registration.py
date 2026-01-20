import open3d as o3d
import numpy as np

def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    """
    This is the function that students will implement.
    It should take two open3d.geometry.PointCloud objects as input
    and return a 4x4 numpy array representing the transformation matrix
    that aligns pcd2 to pcd1.
    """
    # Placeholder implementation: returns an identity matrix
    return np.identity(4)
