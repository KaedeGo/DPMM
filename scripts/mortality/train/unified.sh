CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type unified \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/unified_partial

# nohup sh scripts/mortality/train/unified.sh > ihm_unified_partial.log 2>&1 &