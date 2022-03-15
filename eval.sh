CUDA_VISIBLE_DEVICES=0 python3 eval.py --type static --split habitat --pretrained_path \
/scratch/jaidev/models/habitat/crossView/weights_90/ --data_path ./datasets/odometry/ \
--height 1024 --width 1024 --occ_map_size 256 \
--out_dir /scratch/jaidev/model_outputs/
