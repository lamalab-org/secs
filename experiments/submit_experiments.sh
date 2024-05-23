#!/bin/bash
# List of configuration names
CONFIG_NAMES=("smi_graph", "smi_graph_sf", "smi_sf_fps_graph")

# Submit jobs
for CONFIG in "${CONFIG_NAMES[@]}"; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=molbind_${CONFIG}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100GB

source ~/miniconda3/bin/activate
conda activate molbind

python train.py --config-name ${CONFIG}
EOT
done
