CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python3 train.py --type static --split habitat
--data_path ./datasets/ --model_name habitat_afterfixingdimerror --num_workers 10
