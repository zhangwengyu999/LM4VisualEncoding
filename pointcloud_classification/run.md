# Single GPU

## ScanObjectNN

### ScanObjectNN_models [STOPPED]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA

## ModelNet40

### ModelNet40 1k LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer.yaml PointTransformer_ModelNet40_1k_LLaMA

### ModelNet40 1k NO LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_no_llama.yaml PointTransformer_ModelNet40_1k_NO_LLaMA



### ModelNet40 4k LLaMA [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point.yaml PointTransformer_ModelNet40_4k_LLaMA

### ModelNet40 4k NO LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point_no_llama.yaml PointTransformer_ModelNet40_4k_NO_LLaMA



### ModelNet40 8k LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point.yaml PointTransformer_ModelNet40_8k_LLaMA

### ModelNet40 8k NO LLaMA [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point_no_llama.yaml PointTransformer_ModelNet40_8k_NO_LLaMA




# Slurm
sbatch scripts/train_slurm.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA