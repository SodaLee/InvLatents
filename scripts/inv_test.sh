export WANDB_ENABLE=0
NCCL_ASYNC_ERROR_HANDLING=0 \
deepspeed --include localhost:0,3,6,7 finetune_sdm_yaml.py \
--cf config/tm_inv/yz_tiktok_S256L16_xformers_tsv_temdisco_temp_attn.py \
--eval_visu --root_dir /data1/lihaochen/DisCo_run \
--local_train_batch_size 4 \
--local_eval_batch_size 4 \
--log_dir logs/tiktok_inv_refine \
--epochs 20 --deepspeed \
--eval_step 2000 --save_step 2000 \
--gradient_accumulate_steps 2 \
--learning_rate 1e-4 --fix_dist_seed --loss_target "noise" \
--train_yaml /data1/lihaochen/TikTok_finetuning/composite_offset/train_TiktokDance-poses-masks.yaml \
--val_yaml /data1/lihaochen/TikTok_finetuning/composite_offset/new10val_TiktokDance-poses-masks.yaml \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--train_sample_interval 4 \
--nframe 8 \
--frame_interval 1 \
--conds "poses" "masks" \
--densepose \
--pretrained_model_path /data1/lihaochen/DisCo_run/sd-image-variations-diffusers \
--pretrained_model /data1/lihaochen/DisCo_run/logs/tiktok_posecombine_try0/last.pth/mp_rank_00_model_states.pt \
--guidance_scale 1.5 \
--eval_save_filename test
