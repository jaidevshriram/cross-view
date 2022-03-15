CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 train.py --type static --split habitat \
--data_path ./datasets/ --model_name habitat_rgb_1024_256_19Jan_weight5 --num_workers 10 \
--static_weight 5
