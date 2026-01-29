import argparse
import itertools
import subprocess
import sys
import pandas as pd
import os
import concurrent.futures
import time

# --- Helper Functions (Must be at module level for pickle) ---

def run_experiment(config_path, params, run_id):
    """
    Runs a single experiment in a subprocess.
    """
    cmd = [sys.executable, "scripts/train.py", "--config", config_path]
    
    # Add overrides
    for key, val in params.items():
        cmd.extend(["--set", f"{key}={val}"])
    
    # Run name suffix
    suffix = f"_gs{run_id}"
    cmd.extend(["--set", f"run.name_suffix={suffix}"])
    
    try:
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        mse_raw = None
        mse_norm = None
        
        if result.returncode != 0:
            print(f"[{run_id}] FAILED. Params: {params}")
            # Uncomment to debug
            # print(result.stderr[-500:])
            return {"mse_raw": None, "mse_norm": None}
        
        # Parse output
        for line in result.stdout.splitlines():
            if "[Done] best_val_mse_raw" in line:
                try:
                    mse_raw = float(line.split("=")[1].strip())
                except: pass
            if "[Done] best_val_mse_norm" in line:
                try:
                    mse_norm = float(line.split("=")[1].strip())
                except: pass
                
        if mse_raw is not None:
             print(f"[{run_id}] FINISHED. MSE={mse_raw:.5f}")
        else:
             print(f"[{run_id}] FINISHED (No Metric).")

        return {"mse_raw": mse_raw, "mse_norm": mse_norm}
        
    except Exception as e:
        print(f"[{run_id}] EXCEPTION: {e}")
        return {"mse_raw": None, "mse_norm": None}

def _worker_func(arg_tuple):
    """
    Worker wrapper for ProcessPoolExecutor.
    """
    idx, params, config_path = arg_tuple
    metrics = run_experiment(config_path, params, idx)
    res_row = params.copy()
    res_row.update(metrics)
    return res_row

# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel Grid Search Runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="grid_search_results.csv")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()
    
    # === GRID SEARCH SPACE (ODE-RNN Default) ===
    # This script is primarily used for ODE-RNN tuning in the context of this project.
    # For other models, users should use specific scripts or modify this space.
    search_space = {
        "model.hidden_dim": [32, 64, 128],
        "model.ode_hidden": [32, 64, 128],
        "train.lr": [0.001, 0.005, 0.01],
        "model.dropout": [0.0, 0.1, 0.2]
    }
    # ===========================================
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    print(f"Starting Parallel Grid Search (Workers={args.workers}). Total combinations: {len(combinations)}")
    print(f"Search Space: {search_space}")
    
    # Prepare arguments for map
    tasks = []
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        tasks.append((i, params, args.config))
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # submit all
        futures = [executor.submit(_worker_func, t) for t in tasks]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res = future.result()
            results.append(res)
            
            # Incremental save
            if i % 5 == 0:
                pd.DataFrame(results).to_csv(args.output, index=False)
                
    total_time = time.time() - start_time
    print(f"\nGrid Search Complete in {total_time:.1f}s. Results saved to {args.output}")
    
    # Analysis
    df = pd.DataFrame(results)
    if not df.empty and "mse_raw" in df.columns:
        df_succ = df[pd.to_numeric(df["mse_raw"], errors='coerce').notnull()]
        if not df_succ.empty:
            best = df_succ.sort_values("mse_raw").iloc[0]
            print("\n🏆 Top Config:")
            print(best)
        else:
            print("No successful runs found.")

if __name__ == "__main__":
    main()
