exp_folder=$1
pred_folder="${2:-${exp_folder}/pred_gs1.5_scale-cond1.0-ref1.0}"
gt_folder=${3:-${exp_folder}/gt}

echo ${pred_folder}
echo ${gt_folder}
# # L1 SSIM LPIPS and PSNR
python  tool/metrics/metric_center.py --root_dir /data1/lihaochen/DisCo_run --path_gen ${pred_folder}/ --path_gt ${gt_folder}/ --type l1 ssim lpips psnr  clean-fid --write_metric_to ${exp_folder}/metrics_l1_ssim_lpips_psnr.json

# Pytorch-FID
python -m pytorch_fid ${pred_folder}/ ${gt_folder}/ --device cuda:4

# # generate MP4s
# python tool/video/yz_gen_gifs_for_fvd_subfolders.py -i ${pred_folder} -o ${pred_folder}_16framemp4 --fps 3 --format mp4
# python tool/video/yz_gen_gifs_for_fvd_subfolders.py -i ${gt_folder} -o ${gt_folder}_16framemp4 --fps 3 --format mp4

#  FVD eval
# the root dir should be the dir containing the fvd pretrain model (resnet-50-kinetics and i3d)
python  tool/metrics/metric_center.py --root_dir /data1/lihaochen/DisCo_run --path_gen ${pred_folder}/ --path_gt ${gt_folder}/ --type fid-vid fvd  --write_metric_to ${exp_folder}/metrics_fid-vid_fvd.json --number_sample_frames 16 --sample_duration 16

# python tool/video/yz_gen_vid_subfolders.py -i ${pred_folder} --interval 16
