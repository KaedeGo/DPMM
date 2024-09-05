CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 50 --batch_size 32 --lr 6.493e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--fusion_type daft --layer_after 4 \
--labels_set mortality \
--task in-hospital-mortality \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/daft_partial

# nohup sh scripts/mortality/train/daft.sh > ihm_daft_partial.log 2>&1 &