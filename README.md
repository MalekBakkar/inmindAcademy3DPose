# Point Cloud Registration Assignment

## Objective

The goal of this assignment is to implement a point cloud registration algorithm. You will be provided with two point clouds: a source and a target. Your task is to find the rigid transformation (rotation and translation) that aligns the source point cloud to the target point cloud.

## Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/malek-wahidi/AcademyPointClouds.git
    cd AcademyPointClouds
    ```

2.  **Set up the environment:**
    This project uses `uv` for package management.
    ```bash
    uv sync
    ```

3.  **Implement your registration function:**
    Open the `registration.py` file. You will find a function called `register`. You need to implement your registration algorithm inside this function. The function takes two `open3d.geometry.PointCloud` objects as input and should return a 4x4 NumPy array representing the transformation matrix.

4.  **Run the code:**
    You can run the main script to test your implementation:
    ```bash
    uv run main.py
    ```
    This will load a sample point cloud, create a transformed version of it, and then call your `register` function to find the alignment. The script will also time how long your registration takes.

5.  **Submit your solution:**
    Create a Pull Request to the main repository with your changes in `registration.py` *only*. Your submission will be evaluated based on:
    *   **Correctness:** How well your algorithm aligns the point clouds.
    *   **Performance:** The execution time of your registration function.
    *   **Code Quality:** The clarity, and efficiency of your implementation.

## Troubleshooting

If you encounter errors related to Open3D visualization (such as `GLFW Error: Wayland: The platform does not support setting the window position` or `Failed creating OpenGL window`) on Linux systems using Wayland, try running the script with X11 compatibility:

```bash
XDG_SESSION_TYPE=x11 uv run main.py
```

This can help resolve issues with OpenGL window creation in some desktop environments.