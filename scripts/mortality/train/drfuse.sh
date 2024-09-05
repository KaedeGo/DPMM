CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--task in-hospital-mortality \
--fusion_type drfuse --layer_after 4 \
--labels_set mortality \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/drfuse_partial

# nohup sh scripts/mortality/train/drfuse.sh > ihm_drfuse_partial.log 2>&1 &