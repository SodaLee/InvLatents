# Latent Inversion for Consistent Identity Preservation in Character Animation

The overall network structure code is located in `config/tm_inv/tem_disco_temp_attn/net_combine.py`, and the feature injection module can be found in `register.py` in the same directory.

## Installation
### Environment Setup
This repository has been tested on the following platform:
Python 3.11.9, PyTorch 2.5.1 with CUDA 12.4 and cuDNN 9.1, Ubuntu 22.04.4

To clone the repo, run:
```
git clone https://github.com/SodaLee/InvLatents.git
```
Next, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 
```
conda env create -f environment.yml -n py311
conda activate py311

## for acceleration
pip install --user deepspeed
pip install -v -U git+ssh://git@github.com/facebookresearch/xformers.git@main#egg=xformers

## you may need to downgrade prototbuf to 3.20.x
pip install protobuf==3.20.0
```

## TikTok dataset
We use the [TikTok dataset](https://www.yasamin.page/hdnet_tiktok) for the fine-tuning. 

We have already pre-processed the tiktok data with the efficient TSV format which can be downloaded **[here (Google Drive)](https://drive.google.com/file/d/1_b4naNB1QozGL-tKyHwSSYzTw8RIh5z3/view?usp=sharing)**. (Note that we only use the 1st frame of each TikTok video as the reference image.)

The data folder structure should be like:

```
Data Root
└── composite_offset/
    ├── train_xxx.yaml  # The path need to be then specified in the training args
    └── val_xxx.yaml
    ...
└── TikTokDance/
    ├── xxx_images.tsv
    └── xxx_poses.tsv
    ...
```

*PS: If you want to use your own data resource but with our TSV data structure, please follow [PREPRO.MD](https://github.com/Wangt-CN/DisCo/blob/main/PREPRO.md) for reference. 

## Densepose Preparation
Follow the above mentioned data structure, we generate densepose segmentation.
You can refer to densepose_data/gen_densepose_tsv.py

This procedure used densepose_rcnn_R_101_FPN_DL_WC1M_s1x model, please follow [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) for reference.

We also provide our processed densepose test data. [Processed Test Data](https://drive.google.com/file/d/1L-2Ii3cGnO-4bwBIJ_ggCmRrMmVlLhwd/view?usp=sharing)
Please overwrite the contents in the corresponding folder within the Tiktok dataset directory.

## Pre-trained Models
**Pre-trained Model Checkpoint(1.2M): [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/tan317_e_ntu_edu_sg/EoH8KHplKPhGrIdKN6sPx_ABpurpPjNAvU3KdFgaPwNfJQ)**

**Video-finetuned Checkpoint: [Google](https://drive.google.com/file/d/1yHLWAf36Fp9mPsczHWvLgOETszlZMzEx/view?usp=sharing)**

## Training
1. Download the `sd-image-variations-diffusers` from official [diffusers repo](https://huggingface.co/lambdalabs/sd-image-variations-diffusers) and put it according to the config file `pretrained_model_path`. Or you can also choose to modify the `pretrained_model_path`.

2. You may need to download the [pre-trained vision model](https://drive.google.com/file/d/1J8w3fGj6H6kmcW9G8Ff6tRQofblaG5Vn/view?usp=sharing) and revise the path in `gen_eval.sh` for achieving fvd metric.
```
## motion control training
./scripts/stage2.sh
## video clip training
./scripts/stage2_finetune.sh

## an example of the config file, you may need to change the yaml path in the script
export WANDB_ENABLE=0
AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 \
deepspeed --include localhost:0,1,2,3 finetune_sdm_yaml.py --cf config/stage2_controlnet/tiktok_S256L16_xformers_tsv.py \
--do_train --root_dir /data1/lihaochen/DisCo_run \
--local_train_batch_size 64 \
--local_eval_batch_size 64 \
--log_dir logs/tiktok_stage2_try0 \
--epochs 20 --deepspeed \
--eval_step 2000 --save_step 2000 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--train_yaml /data1/lihaochen/TikTok_finetuning/composite_offset/train_TiktokDance-poses-masks.yaml \
--val_yaml /data1/lihaochen/TikTok_finetuning/composite_offset/new10val_TiktokDance-poses-masks.yaml \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--conds "poses" "masks" \
--pretrained_model_path /data1/lihaochen/DisCo_run/sd-image-variations-diffusers \
--stage1_pretrain_path /data1/lihaochen/DisCo_run/pretrain/stage1.pt \
--drop_ref 0.05 \
--guidance_scale 1.5
```

## Inference
```
## latent inversion inference
./scripts/inv_test.sh
```
