
import os
import torch
import numpy as np
from argparse import ArgumentParser
from sklearn.cluster import DBSCAN
from plyfile import PlyData, PlyElement
from scene.gaussian_model import GaussianModel

def cluster_gaussians(model_path, iteration, eps=0.1, min_samples=10, threshold=0.1):
    print(f"Loading model from {model_path} at iteration {iteration}")
    
    # Load the PLY file
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: Could not find PLY file at {ply_path}")
        return

    plydata = PlyData.read(ply_path)
    vertices = plydata.elements[0]
    
    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1)
    refl = np.asarray(vertices["refl"])
    
    # Filter for reflective Gaussians
    mask = refl > threshold
    reflective_xyz = xyz[mask]
    
    if len(reflective_xyz) == 0:
        print("No reflective Gaussians found with the given threshold.")
        return

    print(f"Clustering {len(reflective_xyz)} reflective Gaussians...")
    
    # DBSCAN Clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(reflective_xyz)
    labels = clustering.labels_
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {num_clusters} clusters. (Points marked -1 are noise)")
    
    # Create an output PLY for visualization
    # We'll map labels to colors for easy viewing
    full_labels = np.ones(len(xyz), dtype=np.int32) * -1
    full_labels[mask] = labels
    
    # Generate random colors for clusters
    unique_labels = set(labels)
    colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    label_to_color[-1] = [50, 50, 50] # Noise/Non-reflective is grey
    
    out_colors = np.array([label_to_color[l] for l in full_labels], dtype=np.uint8)
    
    # Save a visualization PLY
    # We keep only XYZ and add Color
    out_vertex = np.empty(len(xyz), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    out_vertex['x'] = xyz[:, 0]
    out_vertex['y'] = xyz[:, 1]
    out_vertex['z'] = xyz[:, 2]
    out_vertex['red'] = out_colors[:, 0]
    out_vertex['green'] = out_colors[:, 1]
    out_vertex['blue'] = out_colors[:, 2]
    
    os.makedirs(os.path.join(model_path, "clusters"), exist_ok=True)
    out_path = os.path.join(model_path, "clusters", f"visualization_{iteration}.ply")
    PlyData([PlyElement.describe(out_vertex, 'vertex')]).write(out_path)
    
    # Also save a numpy file with the mapping for later use
    np_path = os.path.join(model_path, "clusters", f"labels_{iteration}.npy")
    np.save(np_path, full_labels)
    
    print(f"Visualization saved to: {out_path}")
    print(f"Labels saved to: {np_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Cluster reflective Gaussians for 3DGS-DR")
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--iteration", "-i", default=30000, type=int)
    parser.add_argument("--eps", default=0.05, type=float, help="DBSCAN epsilon (spatial radius)")
    parser.add_argument("--min_samples", default=20, type=int, help="DBSCAN min samples per cluster")
    parser.add_argument("--threshold", default=0.01, type=float, help="Reflection strength threshold")
    
    args = parser.parse_args()
    cluster_gaussians(args.model_path, args.iteration, args.eps, args.min_samples, args.threshold)
