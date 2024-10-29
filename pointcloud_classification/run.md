# Single GPU

## ScanObjectNN_models [STOPPED]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA

## ModelNet40 1k LLaMA [RUNNING]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer.yaml PointTransformer_ModelNet40_1k_LLaMA

## ModelNet40 1k NO LLaMA [RUNNING]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_no_llama.yaml PointTransformer_ModelNet40_1k_NO_LLaMA

# Slurm
sbatch scripts/train_slurm.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA