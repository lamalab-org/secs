#!/bin/bash -x

#SBATCH --account hai_molbind # specify the account name, otherwise the job will not run
#SBATCH --time 24:00:00       # specify the running time; on develbooster it is limited to 2 hours
#SBATCH --cpus-per-task 4     # do not change
#SBATCH --ntasks-per-node 4   # number of tasks per node; each node has 4 GPUs
#SBATCH --nodes 2             # the number of nodes; if you request 4 nodes you get 16 GPUs
#SBATCH --partition booster   # the walltime on develbooster is 2 hours, on booster 24 hours; maximum number of nodes on develbooster is 4 nodes
#SBATCH --gres gpu:4          # do not change

##SBATCH --nodelist=jwb[0001-0012,0022-0025,0027-0035,0037-0044,0053-0057,0060-0074,0076,0085-0095,0097-0108,0117-0126,0128-0129,0131-0140,0150-0172,0181-0185,0187-0192,0194-0198,0200,0202-0204,0214-0226,0228-0231,0233-0236,0245-0252,0254-0261,0263-0265,0267-0268,0277-0286,0288-0293,0295,0297-0300,0309-0332,0341-0358,0360,0362-0364,0373-0380,0382-0390,0392-0396,0405-0417,0419-0428,0437-0448,0450-0453,0455-0460,0469-0473,0475,0477-0492,0501,0503-0524,0533-0541,0543-0546,0548-0549,0551-0556,0565-0588,0597-0606,0608-0616,0618-0620,0629-0633,0635-0644,0647-0652,0661-0674,0677-0679,0681-0684,0693-0711,0713-0716,0725-0747,0757-0780,0789-0800,0802-0812,0821-0823,0825-0844,0853-0876,0885-0894,0896-0899,0901-0908,0917-0926,0928-0934,0936-0940,0949-0970,0972,0981-1004,1013-1021,1024-1028,1030-1036,1045-1046,1048-1056,1058-1063,1065-1066,1068,1077-1100,1109-1115,1118-1124,1127-1132,1141-1155,1157-1164,1173-1196,1205-1212,1216,1220,1222,1226,1228,1237]                                                 # do not change
##SBATCH --exclude=jwb0182,jwb0188,jwb0459,jwb0101,jwb0102,jwb0386,jwb0824,jwb0199,jwb0232,jwb1213,jwb0801,jwb0547,jwb0900,jwb0361,jwb0646,jwb0059,jwb0294,jwb0927,jwb0502,jwb0645,jwb1239,jwb0675,jwb0634,jwb0287,jwb0296,jwb0971,jwb0096,jwb1064,jwb0454,jwb1125,jwb1023,jwb1245,jwb0036,jwb0227,jwb1247,jwb0186,jwb1047,jwb0449,jwb1246,jwb1244,jwb0193,jwb1057,jwb0474,jwb0895,jwb1117,jwb0381,jwb1240,jwb0935,jwb0130,jwb1215,jwb0476,jwb0213,jwb0058,jwb1243,jwb1214,jwb0542,jwb0266,jwb0026,jwb1156,jwb0607,jwb0075,jwb0021,jwb0418,jwb1126,jwb1067,jwb0359,jwb0253,jwb1242,jwb0748,jwb0149,jwb0550,jwb0391,jwb1029,jwb0680,jwb1248,jwb1241,jwb1218,jwb1116,jwb1022,jwb0712,jwb0617,jwb0201,jwb0127,jwb1238,jwb1227,jwb1225,jwb1224,jwb1223,jwb1221,jwb1219,jwb1217,jwb0676,jwb0262

######################## DO NOT CHANGE THE BELOW ON JUWELS################################

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
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
srun --cpu-bind=none bash -c "
    export CUDA_VISIBLE_DEVICES='0,1,2,3'
    export PYTHONPATH=''
    bash -c '
        source /p/project/hai_molbind/miniconda3/bin/activate
        conda activate molbind
        python3.12 train.py --config-name sf_graph_fps_frag_smi_struct_1024_linear.yaml
    '
"
########################################################################################