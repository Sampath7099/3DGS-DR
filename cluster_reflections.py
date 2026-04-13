import os
import torch
import numpy as np
from argparse import ArgumentParser
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement

def cluster_gaussians(model_path, iteration, k=12):
    print(f"Loading model from {model_path} at iteration {iteration}")
    
    # Load the PLY file
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: Could not find PLY file at {ply_path}")
        return

    print("Reading PLY data...")
    plydata = PlyData.read(ply_path)
    vertices = plydata.elements[0]
    
    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1)
    
    num_points = len(xyz)
    print(f"Clustering ALL {num_points} Gaussians into {k} clusters using K-Means based on XYZ only...")
    
    # K-Means Clustering
    # Increase n_init for better global search as requested for better separation
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300).fit(xyz)
    labels = kmeans.labels_
    
    # Report cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Cluster sizes:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} points ({(count/num_points)*100:.1f}%)")
    
    print(f"Clustering complete.")
    
    # Generate random colors for clusters
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(k, 3))
    label_to_color = {i: colors[i] for i in range(k)}
    
    out_colors = np.array([label_to_color[l] for l in labels], dtype=np.uint8)
    
    # Save a visualization PLY
    print("Saving visualization...")
    out_vertex = np.empty(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    out_vertex['x'] = xyz[:, 0]
    out_vertex['y'] = xyz[:, 1]
    out_vertex['z'] = xyz[:, 2]
    out_vertex['red'] = out_colors[:, 0]
    out_vertex['green'] = out_colors[:, 1]
    out_vertex['blue'] = out_colors[:, 2]
    
    clusters_dir = os.path.join(model_path, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)
    out_path = os.path.join(clusters_dir, f"visualization_{iteration}.ply")
    PlyData([PlyElement.describe(out_vertex, 'vertex')]).write(out_path)
    
    # Also save a numpy file with the mapping for later use
    np_path = os.path.join(clusters_dir, f"labels_{iteration}.npy")
    np.save(np_path, labels)
    
    print(f"Visualization saved to: {out_path}")
    print(f"Labels saved to: {np_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Cluster ALL Gaussians for 3DGS-DR using K-Means")
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--iteration", "-i", default=30000, type=int)
    parser.add_argument("--k", "-k", default=12, type=int, help="Number of clusters")
    
    args = parser.parse_args()
    cluster_gaussians(args.model_path, args.iteration, args.k)
