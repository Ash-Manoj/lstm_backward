# -*- coding: utf-8 -*-
import os
import pandas as pd
from glob import glob

# List all seed_* directories
seed_dirs = sorted(glob('seed_*'))

# Initialize lists to collect DataFrames
y_sim_list = []
y_sim_ungauged_list = []

# Reference shapes and column/index info (separately for each type)
ref_y_sim_shape = ref_y_sim_columns = ref_y_sim_index = None
ref_yu_shape = ref_yu_columns = ref_yu_index = None

for seed_dir in seed_dirs:
    y_sim_path = os.path.join(seed_dir, 'y_sim.csv')
    y_sim_ungauged_path = os.path.join(seed_dir, 'y_sim_ungauged.csv')

    if not os.path.exists(y_sim_path) or not os.path.exists(y_sim_ungauged_path):
        print(f"?? Missing one or both files in {seed_dir}")
        continue

    try:
        # Read both files
        y_sim = pd.read_csv(y_sim_path, index_col=0)
        yu = pd.read_csv(y_sim_ungauged_path, index_col=0)

        # Check y_sim structure
        if ref_y_sim_shape is None:
            ref_y_sim_shape = y_sim.shape
            ref_y_sim_columns = list(y_sim.columns)
            ref_y_sim_index = y_sim.index.tolist()
        else:
            assert y_sim.shape == ref_y_sim_shape, f"Shape mismatch in {y_sim_path}"
            assert list(y_sim.columns) == ref_y_sim_columns, f"Column mismatch in {y_sim_path}"
            assert y_sim.index.equals(pd.Index(ref_y_sim_index)), f"Index mismatch in {y_sim_path}"

        # Check y_sim_ungauged structure
        if ref_yu_shape is None:
            ref_yu_shape = yu.shape
            ref_yu_columns = list(yu.columns)
            ref_yu_index = yu.index.tolist()
        else:
            assert yu.shape == ref_yu_shape, f"Shape mismatch in {y_sim_ungauged_path}"
            assert list(yu.columns) == ref_yu_columns, f"Column mismatch in {y_sim_ungauged_path}"
            assert yu.index.equals(pd.Index(ref_yu_index)), f"Index mismatch in {y_sim_ungauged_path}"

        # Append data
        y_sim_list.append(y_sim)
        y_sim_ungauged_list.append(yu)

    except Exception as e:
        print(f"? Error processing {seed_dir}: {e}")

# Compute averages and save if data was collected
if y_sim_list:
    y_sim_avg = sum(y_sim_list) / len(y_sim_list)
    os.makedirs('seed_average', exist_ok=True)
    y_sim_avg.to_csv('seed_average/y_sim.csv')
    print("? Averaged y_sim saved.")
else:
    print("? No valid y_sim data found.")

if y_sim_ungauged_list:
    y_sim_ungauged_avg = sum(y_sim_ungauged_list) / len(y_sim_ungauged_list)
    os.makedirs('seed_average', exist_ok=True)
    y_sim_ungauged_avg.to_csv('seed_average/y_sim_ungauged.csv')
    print("? Averaged y_sim_ungauged saved.")
else:
    print("? No valid y_sim_ungauged data found.")
