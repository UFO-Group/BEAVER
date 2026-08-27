#!/bin/bash
#SBATCH -J lgbm_grid1
#SBATCH -p gpu
#SBATCH -w gpu3
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --error=/home/fangxr/ML_Tool/2.ML/grid/task/rf_melt-%J.err
#SBATCH --output=/home/fangxr/ML_Tool/2.ML/grid/task/rf_melt-%J.out

export CUDA_VISIBLE_DEVICES=1

cd "$SLURM_SUBMIT_DIR"

python /home/fangxr/ML_Tool/2.ML/grid/grid_search_lgbm.py \
  --in_csv "/home/fangxr/ML_Tool/2.ML/DataBase/0401/10/MeltingTemperature-pooled-4096.csv" \
  --target_col "Melting Temperature (°C)" \
  --cv_folds 10 \
  --n_jobs 8 \
  --grid_preset small \
  --save_model \
  --out_dir "/home/fangxr/ML_Tool/2.ML/grid/outputs/MeltingTemperature/MeltingTemperature-lgbm_loop"
  
# "Melting Temperature (°C)"
# "Glass Transition Temperature (°C)"
# "Elongation at Break (%) log10"
# "Young's Modulus (kPa) log10"
# "Tensile Strength (MPa) log10"

# "MeltingTemperature")
# "GlassTransitionTemperature")
# "ElongationBreak")
# "YoungsModulus")
# "TensileStrength")
