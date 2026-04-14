import argparse
import subprocess
import os
import re
import csv
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Log 3DGS hyperparameters and evaluation metrics to CSV.")
    parser.add_argument("--model_path", required=True, help="Path to the model output folder (e.g., output/my_30k_run)")
    parser.add_argument("--notes", default="", help="Any custom notes for this run (e.g., 'Aggressive normal test')")
    return parser.parse_args()

def extract_hyperparams():
    params = {}
    
    # 1. From arguments/__init__.py
    try:
        with open("arguments/__init__.py", "r") as f:
            content = f.read()
            # Extract ALL 'self.key = value' arguments to ensure we capture live edits
            matches = re.findall(r"self\.([a-zA-Z0-9_]+)\s*=\s*([^\n]+)", content)
            for key, val in matches:
                # Clean up comments and underscores for precise logging
                clean_val = val.split('#')[0].strip()
                clean_val = clean_val.replace("_", "") # e.g. 24_000 -> 24000
                params[key] = clean_val
    except Exception as e:
        print(f"Warning: Could not read arguments/__init__.py: {e}")

    # 2. From scene/gaussian_model.py
    try:
        with open("scene/gaussian_model.py", "r") as f:
            content = f.read()
            # Extract thresholds from specific functions using a local window search
            funcs = ["dist_rot", "dist_color", "enlarge_refl_scales", 
                     "enlarge_refl_scales_strategy2", "reset_opacity0", 
                     "reset_opacity1", "reset_opacity1_strategy2"]
            for func in funcs:
                func_idx = content.find(f"def {func}(")
                if func_idx != -1:
                    window = content[func_idx:func_idx+800] # look 800 chars ahead
                    for var in ["REFL_MSK_THR", "DIST_RANGE", "ENLARGE_SCALE", "RESET_V", "RESET_B"]:
                        m = re.search(fr"{var}\s*=\s*([\d\.\-]+)", window)
                        if m: 
                            params[f"{func}_{var}"] = m.group(1)
            
    except Exception as e:
        print(f"Warning: Could not read scene/gaussian_model.py: {e}")
        
    return params

def run_evaluation(model_path):
    print(f"Running eval.py on {model_path} to gather metrics...")
    try:
        # Run eval.py and capture the stdout
        result = subprocess.run(
            ["python", "eval.py", "--model_path", model_path],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        
        metrics = {}
        # Parse the output line, e.g., psnr:26.561,ssim:0.934,lpips:0.065,fps:169.5
        m = re.search(r"psnr:([\d\.]+),ssim:([\d\.]+),lpips:([\d\.]+),fps:([\d\.]+)", output)
        if m:
            metrics["psnr"] = float(m.group(1))
            metrics["ssim"] = float(m.group(2))
            metrics["lpips"] = float(m.group(3))
            metrics["fps"] = float(m.group(4))
        else:
            print("Warning: Could not parse eval.py output! Ensure eval.py ran successfully.")
            print(output)
        return metrics
    except subprocess.CalledProcessError as e:
        print("Error evaluating model! eval.py failed.")
        print(e.stderr)
        return {}

def main():
    args = parse_args()
    
    # 1. Scrape the current source code state
    params = extract_hyperparams()
    
    # 2. Run evaluation
    metrics = run_evaluation(args.model_path)
    
    if not metrics:
        print("Aborting log due to missing evaluation metrics.")
        return
        
    # 3. Prepare the CSV row
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path.split("/")[-1], # Just save the folder name
        "notes": args.notes,
        "psnr": metrics.get("psnr", ""),
        "ssim": metrics.get("ssim", ""),
        "lpips": metrics.get("lpips", ""),
        "fps": metrics.get("fps", ""),
        "lambda_refl_smooth": params.get("lambda_refl_smooth", ""),
        "normal_prop_iter": params.get("normal_prop_until_iter", ""),
        "feature_rest_iter": params.get("feature_rest_from_iter", ""),
        "rotation_lr": params.get("rotation_lr", ""),
        "dist_rot_REFL_THR": params.get("dist_rot_REFL_MSK_THR", ""),
        "dist_color_REFL_THR": params.get("dist_color_REFL_MSK_THR", ""),
        "enlarge_SCALE": params.get("enlarge_ENLARGE_SCALE", "")
    }
    
    csv_file = "hyperparameter_experiments.csv"
    file_exists = os.path.isfile(csv_file)
    
    fieldnames = list(row.keys())
    
    # 4. Append to CSV
    with open(csv_file, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    print(f"\n==========================================")
    print(f"Experiment Logged Successfully!")
    print(f"PSNR: {row['psnr']:.3f} | SSIM: {row['ssim']:.3f} | LPIPS: {row['lpips']:.3f}")
    print(f"Saved to:  {csv_file}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
