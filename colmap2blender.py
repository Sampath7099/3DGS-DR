import os
import sys
import numpy as np
import json
import struct
import collections
from PIL import Image
from plyfile import PlyData, PlyElement

# --- Self-contained COLMAP Loading logic ---

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class ImageWrapper(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_intrinsics_binary(path):
    CAMERA_MODELS = {
        0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
        4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
        8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE"
    }
    MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id, width, height = camera_properties
            model_name = CAMERA_MODELS[model_id]
            num_params = MODEL_PARAMS[model_id]
            params = read_next_bytes(fid, 8*num_params, "d"*num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
    return cameras

def read_extrinsics_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id, qvec, tvec, camera_id = binary_image_properties[0], np.array(binary_image_properties[1:5]), np.array(binary_image_properties[5:8]), binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2D) # skip points2D
            images[image_id] = ImageWrapper(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=None, point3D_ids=None)
    return images

def read_points3D_binary(path):
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            xyzs[p_id] = np.array(binary_point_line_properties[1:4])
            rgbs[p_id] = np.array(binary_point_line_properties[4:7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.read(8 * track_length) # skip track
    return xyzs, rgbs, None

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# --- Main Conversion Logic ---

def colmap2blender(source_path):
    print(f"Converting COLMAP dataset at {source_path} to Blender format...")
    
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(source_path, "sparse/0/images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(source_path, "sparse/0/cameras.bin"))
    except Exception as e:
        print(f"Error loading COLMAP data: {e}")
        return

    # identify image folder - prioritize images_8
    img_folder = os.path.join(source_path, "images_8")
    if not os.path.exists(img_folder):
        img_folder = os.path.join(source_path, "images")
    
    if not os.path.exists(img_folder):
        # check for other folders like images_4
        for f in os.listdir(source_path):
            if f.startswith("images") and os.path.isdir(os.path.join(source_path, f)):
                img_folder = os.path.join(source_path, f)
                break
    
    print(f"Using images from: {img_folder}")

    keys = sorted(cam_extrinsics.keys(), key=lambda x: cam_extrinsics[x].name)
    test_keys = [k for i, k in enumerate(keys) if i % 8 == 0]
    train_keys = [k for i, k in enumerate(keys) if i % 8 != 0]

    for split, keys_to_use in [("train", train_keys), ("test", test_keys)]:
        print(f"Processing {split} set ({len(keys_to_use)} images)...")
        split_dir = os.path.join(source_path, split)
        os.makedirs(split_dir, exist_ok=True)
        
        frames = []
        camera_angle_x = 0
        
        for k in keys_to_use:
            extr = cam_extrinsics[k]
            intr = cam_intrinsics[extr.camera_id]
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)
            W2C = np.eye(4)
            W2C[:3, :3] = R
            W2C[:3, 3] = T
            C2W = np.linalg.inv(W2C)
            C2W_blender = C2W.copy()
            C2W_blender[:3, 1:3] *= -1 
            
            focal_x = intr.params[0]
            camera_angle_x = 2 * np.arctan(intr.width / (2 * focal_x))
            
            img_name = os.path.basename(extr.name)
            dest_img_name = os.path.splitext(img_name)[0] + ".png"
            dest_img_path = os.path.join(split_dir, dest_img_name)
            
            if not os.path.exists(dest_img_path):
                img = Image.open(os.path.join(img_folder, img_name))
                img = img.convert("RGBA")
                img.save(dest_img_path)
            
            frames.append({
                "file_path": f"./{split}/{os.path.splitext(img_name)[0]}",
                "transform_matrix": C2W_blender.tolist()
            })
        
        transforms = {
            "camera_angle_x": camera_angle_x,
            "frames": frames
        }
        
        json_out = os.path.join(source_path, f"transforms_{split}.json")
        with open(json_out, "w") as f:
            json.dump(transforms, f, indent=4)
        print(f"Wrote {json_out}")

    ply_out = os.path.join(source_path, "points3d.ply")
    if not os.path.exists(ply_out):
        print("Generating points3d.ply...")
        bin_path = os.path.join(source_path, "sparse/0/points3D.bin")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_out, xyz, rgb)
            print(f"Wrote {ply_out}")
        except Exception as e:
            print(f"Error generating PLY: {e}")

    print("Finished conversion.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python colmap2blender.py <path_to_dataset>")
    else:
        colmap2blender(sys.argv[1])
