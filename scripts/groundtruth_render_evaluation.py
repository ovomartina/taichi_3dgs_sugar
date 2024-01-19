from decimal import MIN_EMIN
from tokenize import String
import numpy as np 
import open3d as o3d
import os
import pandas as pd
from matplotlib import pyplot as plt

def point_cloud_groundtruth_comparison(groundtruth_file_path, pointcloud_reconstruction_path, show_pointclouds=False, show_plots=False):
    
    pc_groundtruth = o3d.io.read_point_cloud(groundtruth_file_path)
    pc_reconstruction = o3d.io.read_point_cloud(pointcloud_reconstruction_path)   
    
    pc_groundtruth.paint_uniform_color([0,0,1])
    pc_reconstruction.paint_uniform_color([0.5,0.5,0])
    
    # The reconstructed mesh contains several outliers; only consider the points closer to the groundtruth mesh
    points = np.asarray(pc_reconstruction.points)
    norms = np.linalg.norm(points, axis=1)
    selected_points = points[norms < 1]
    pc_selected = o3d.geometry.PointCloud() 
    pc_selected.points = o3d.utility.Vector3dVector(selected_points)
    
    # visualization
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    if show_pointclouds:
        downpcd = pc_groundtruth.voxel_down_sample(voxel_size=0.05)
        o3d.visualization.draw_geometries([downpcd, pc_reconstruction, axis, ]) #pc_selected
    
    # Calculate distances of pc_1 to pc_2. 
    dist_pc1_pc2 = pc_groundtruth.compute_point_cloud_distance(pc_selected)

    # dist_pc1_pc2 is an Open3d object, we need to convert it to a numpy array to 
    # acess the data
    dist_pc1_pc2 = np.asarray(dist_pc1_pc2)

    if show_plots:
        # Boxplot, histogram and serie to visualize distances. 
        df = pd.DataFrame({"distances": dist_pc1_pc2}) # transform to a dataframe
        ax1 = df.boxplot(return_type="axes") # BOXPLOT
        ax2 = df.plot(kind="hist", alpha=0.5, bins = 1000) # HISTOGRAM
        ax3 = df.plot(kind="line") # SERIE
        plt.show()
 
    return dist_pc1_pc2

def get_pointlcoud_dimension(pointcloud_file_path):
    pc = o3d.io.read_point_cloud(pointcloud_file_path)
    bounding_box = pc.get_axis_aligned_bounding_box()
    extent = bounding_box.get_extent()  
    return np.max(extent)


def main():
    show_pointclouds = True
    show_plots = False
    groundtruth_file_path = "output/replica/room_0/point_clouds/mesh.pcd"
    
    # Compare all reconstructed pointclouds in the folder
    path = 'output/replica/room_0/point_clouds/'
    fileList = os.listdir(path)
    
    distances = []
    for pointcloud_reconstruction_path in fileList:
        if pointcloud_reconstruction_path.startswith('pointcloud_'):
            print(pointcloud_reconstruction_path)
            distances.append(point_cloud_groundtruth_comparison(groundtruth_file_path, os.path.join(path+pointcloud_reconstruction_path),show_pointclouds, show_plots))
    
    max_extent = get_pointlcoud_dimension(groundtruth_file_path)
    print(max_extent)
    
    distances = np.array(distances).flatten()
    avg_distance = np.mean(distances)
    median_distance = np.median(distances)
    print("Average distance:", avg_distance)
    print("Median distance:", median_distance)
    
    df = pd.DataFrame({"distances": distances}) # transform to a dataframe
    # Some graphs
    ax1 = df.boxplot(return_type="axes") # BOXPLOT
    ax1.set_title("Boxplot")
    ax2 = df.plot(kind="hist", alpha=0.5, bins = 1000) # HISTOGRAM
    ax3 = df.plot(kind="line") # SERIE
    plt.show()


if __name__ == '__main__':
    main()