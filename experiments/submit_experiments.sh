#!/bin/bash
# List of configuration names
CONFIG_NAMES=(smi_graph.yaml smi_graph_sf.yaml smi_sf_fps_graph.yaml)

# Submit jobs
for CONFIG in "${CONFIG_NAMES[@]}"; do
    echo "Submitting job for config: ${CONFIG}"
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=molbind_${CONFIG::-5}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=100GB

source ~/miniconda3/bin/activate
conda activate molbind

python train.py --config-name ${CONFIG}
EOT
done
