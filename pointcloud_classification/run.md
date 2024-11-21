# Single GPU

## ScanObjectNN

### ScanObjectNN_models [STOPPED]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA

## ModelNet40

### ModelNet40 1k LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer.yaml PointTransformer_ModelNet40_1k_LLaMA

CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer.yaml PointTransformer_ModelNet40_1k_LLaMA_test

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer/ModelNet_models/PointTransformer_ModelNet40_1k_LLaMA/ckpt-best.pth
92.4635

### ModelNet40 1k NO LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_no_llama.yaml PointTransformer_ModelNet40_1k_NO_LLaMA

CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_no_llama.yaml PointTransformer_ModelNet40_1k_NO_LLaMA_test

#### eval [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_no_llama.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_no_llama/ModelNet_models/PointTransformer_ModelNet40_1k_NO_LLaMA/ckpt-best.pth
92.9498


### ModelNet40 4k LLaMA [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point.yaml PointTransformer_ModelNet40_4k_LLaMA

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_4096point.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_4096point/ModelNet_models/PointTransformer_ModelNet40_4k_LLaMA/ckpt-best.pth
93.0713

### ModelNet40 4k NO LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point_no_llama.yaml PointTransformer_ModelNet40_4k_NO_LLaMA

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_4096point_no_llama.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_4096point_no_llama/ModelNet_models/PointTransformer_ModelNet40_4k_NO_LLaMA/ckpt-best.pth
92.9903

### ModelNet40 8k LLaMA [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point.yaml PointTransformer_ModelNet40_8k_LLaMA

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_8192point.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_8192point/ModelNet_models/PointTransformer_ModelNet40_8k_LLaMA/ckpt-best.pth
93.7601

### ModelNet40 8k NO LLaMA [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point_no_llama.yaml PointTransformer_ModelNet40_8k_NO_LLaMA

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_8192point_no_llama.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_8192point_no_llama/ModelNet_models/PointTransformer_ModelNet40_8k_NO_LLaMA/ckpt-best.pth
93.5981


## ModelNet40 + Only MLP


### ModelNet40 1k Only MLP 2 [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/ModelNet_models/PointTransformer_only_mlp_2.yaml PointTransformer_ModelNet40_1k_ONLY_MLP_2

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_only_mlp_2.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_only_mlp_2/ModelNet_models/PointTransformer_ModelNet40_1k_ONLY_MLP_2/ckpt-best.pth

### ModelNet40 1k Only MLP 4 [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/ModelNet_models/PointTransformer_only_mlp_4.yaml PointTransformer_ModelNet40_1k_ONLY_MLP_4

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_only_mlp_4.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_only_mlp_4/ModelNet_models/PointTransformer_ModelNet40_1k_ONLY_MLP_4/ckpt-best.pth

### ModelNet40 1k Only MLP 6 [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_only_mlp_6.yaml PointTransformer_ModelNet40_1k_ONLY_MLP_6

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_only_mlp_6.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_only_mlp_6/ModelNet_models/PointTransformer_ModelNet40_1k_ONLY_MLP_6/ckpt-best.pth

### ModelNet40 1k Only MLP 8 [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_only_mlp_8.yaml PointTransformer_ModelNet40_1k_ONLY_MLP_8

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_only_mlp_8.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_only_mlp_8/ModelNet_models/PointTransformer_ModelNet40_1k_ONLY_MLP_8/ckpt-best.pth




### ModelNet40 4k Only MLP 2 [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_2.yaml PointTransformer_ModelNet40_4k_ONLY_MLP_2

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_2.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_4096point_only_mlp_2/ModelNet_models/PointTransformer_ModelNet40_4k_ONLY_MLP_2/ckpt-best.pth

### ModelNet40 4k Only MLP 4 [DONE]
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_4.yaml PointTransformer_ModelNet40_4k_ONLY_MLP_4

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_4.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_4096point_only_mlp_4/ModelNet_models/PointTransformer_ModelNet40_4k_ONLY_MLP_4/ckpt-best.pth

### ModelNet40 4k Only MLP 6 [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_6.yaml PointTransformer_ModelNet40_4k_ONLY_MLP_6

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_6.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_4096point_only_mlp_6/ModelNet_models/PointTransformer_ModelNet40_4k_ONLY_MLP_6/ckpt-best.pth

### ModelNet40 4k Only MLP 8 [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_8.yaml PointTransformer_ModelNet40_4k_ONLY_MLP_8

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_4096point_only_mlp_8.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_4096point_only_mlp_8/ModelNet_models/PointTransformer_ModelNet40_4k_ONLY_MLP_8/ckpt-best.pth




### ModelNet40 8k Only MLP 2 [RUNNING]
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_2.yaml PointTransformer_ModelNet40_8k_ONLY_MLP_2

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_2.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_8192point_only_mlp_2/ModelNet_models/PointTransformer_ModelNet40_8k_ONLY_MLP_2/ckpt-best.pth

### ModelNet40 8k Only MLP 4 [RUNNING]
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_4.yaml PointTransformer_ModelNet40_8k_ONLY_MLP_4

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_4.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_8192point_only_mlp_4/ModelNet_models/PointTransformer_ModelNet40_8k_ONLY_MLP_4/ckpt-best.pth

### ModelNet40 8k Only MLP 6 [RUNNING]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_6.yaml PointTransformer_ModelNet40_8k_ONLY_MLP_6

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_6.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_8192point_only_mlp_6/ModelNet_models/PointTransformer_ModelNet40_8k_ONLY_MLP_6/ckpt-best.pth


### ModelNet40 8k Only MLP 8 [RUNNING]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_8.yaml PointTransformer_ModelNet40_8k_ONLY_MLP_8

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_8192point_only_mlp_8.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_8192point_only_mlp_8/ModelNet_models/PointTransformer_ModelNet40_8k_ONLY_MLP_8/ckpt-best.pth


## ModelNet40 + VGG16

### ModelNet40 1k VGG16 [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_vgg16.yaml PointTransformer_ModelNet40_1k_VGG16

#### eval [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_vgg16.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_vgg16/ModelNet_models/PointTransformer_ModelNet40_1k_VGG16/ckpt-best.pth
92.5851

### ModelNet40 4k VGG16 [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_4096point_vgg16.yaml PointTransformer_ModelNet40_4k_VGG16

#### eval [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_4096point_vgg16.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_4096point_vgg16/ModelNet_models/PointTransformer_ModelNet40_4k_VGG16/ckpt-best.pth
92.7066

### ModelNet40 8k VGG16 [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointTransformer_8192point_vgg16.yaml PointTransformer_ModelNet40_8k_VGG16

#### eval [TODO]
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh configs/ModelNet_models/PointTransformer_8192point_vgg16.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointTransformer_8192point_vgg16/ModelNet_models/PointTransformer_ModelNet40_8k_VGG16/ckpt-best.pth
92.5446

## ModelNet40 + Mamba

### ModelNet40 1k Mamba 1l [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointMamba_1.yaml PointMamba_ModelNet40_1k_Mamba_1l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh configs/ModelNet_models/PointMamba_1.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_1/ModelNet_models/PointMamba_ModelNet40_1k_Mamba_1l/ckpt-best.pth
89.1815

### ModelNet40 1k Mamba 2l [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointMamba_2.yaml PointMamba_ModelNet40_1k_Mamba_2l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh configs/ModelNet_models/PointMamba_2.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_2/ModelNet_models/PointMamba_ModelNet40_1k_Mamba_2l/ckpt-best.pth
90.1945

### ModelNet40 1k Mamba 3l [DONE]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointMamba_3.yaml PointMamba_ModelNet40_1k_Mamba_3l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh configs/ModelNet_models/PointMamba_3.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_3/ModelNet_models/PointMamba_ModelNet40_1k_Mamba_3l/ckpt-best.pth
89.7488



### ModelNet40 4k Mamba 1l [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/ModelNet_models/PointMamba_4096point_1.yaml PointMamba_ModelNet40_4k_Mamba_1l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh configs/ModelNet_models/PointMamba_4096point_1.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_4096point_1/ModelNet_models/PointMamba_ModelNet40_4k_Mamba_1l/ckpt-best.pth
88.9384

### ModelNet40 4k Mamba 2l [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/ModelNet_models/PointMamba_4096point_2.yaml PointMamba_ModelNet40_4k_Mamba_2l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh configs/ModelNet_models/PointMamba_4096point_2.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_4096point_2/ModelNet_models/PointMamba_ModelNet40_4k_Mamba_2l/ckpt-best.pth
89.5057

### ModelNet40 4k Mamba 3l [DONE]
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh configs/ModelNet_models/PointMamba_4096point_3.yaml PointMamba_ModelNet40_4k_Mamba_3l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh configs/ModelNet_models/PointMamba_4096point_3.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_4096point_3/ModelNet_models/PointMamba_ModelNet40_4k_Mamba_3l/ckpt-best.pth
89.1410




### ModelNet40 8k Mamba 1l [DONE]
CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh configs/ModelNet_models/PointMamba_8192point_1.yaml PointMamba_ModelNet40_8k_Mamba_1l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=1 bash scripts/eval.sh configs/ModelNet_models/PointMamba_8192point_1.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_8192point_1/ModelNet_models/PointMamba_ModelNet40_8k_Mamba_1l/ckpt-best.pth
88.6143

### ModelNet40 8k Mamba 2l [DONE]
CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh configs/ModelNet_models/PointMamba_8192point_2.yaml PointMamba_ModelNet40_8k_Mamba_2l

#### eval [DONE]
CUDA_VISIBLE_DEVICES=1 bash scripts/eval.sh configs/ModelNet_models/PointMamba_8192point_2.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_8192point_2/ModelNet_models/PointMamba_ModelNet40_8k_Mamba_2l/ckpt-best.pth
89.2220


### ModelNet40 8k Mamba 3l [RUNNING]
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh configs/ModelNet_models/PointMamba_8192point_3.yaml PointMamba_ModelNet40_8k_Mamba_3l

#### eval [TODO]
CUDA_VISIBLE_DEVICES=1 bash scripts/eval.sh configs/ModelNet_models/PointMamba_8192point_3.yaml /home/wengyu/work/LM4VisualEncoding/pointcloud_classification/experiments/PointMamba_8192point_3/ModelNet_models/PointMamba_ModelNet40_8k_Mamba_3l/ckpt-best.pth


# Slurm
sbatch scripts/train_slurm.sh configs/ScanObjectNN_models/PointLLaMa_hardest.yaml PointTransformer_LLaMA