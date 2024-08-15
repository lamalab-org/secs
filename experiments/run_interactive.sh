export SRUN_CPUS_PER_TASK=4
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
export MASTER_ADDR="${MASTER_ADDR}i"

# Load the necessary modules
module --force purge
module load Stages/2024  GCCcore/.12.3.0

############################RUN A TRAINING SCRIPT WITH SRUN#############################
srun --gres=gpu:4 --nodes=1 --ntasks-per-node=4 --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    export PYTHONPATH=''
    bash -c '
        source /p/project/hai_molbind/miniconda3/bin/activate
        conda activate molbind
        python train.py 'experiment=cnmr_simulated'
    '
"
#########################################################################################