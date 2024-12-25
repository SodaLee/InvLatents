export WANDB_ENABLE=0
AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 \
deepspeed --include localhost:4,5,6,7 finetune_sdm_yaml.py --cf config/stage2_controlnet/tiktok_S256L16_xformers_tsv.py \
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