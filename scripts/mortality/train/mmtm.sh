CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 4.355e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--task in-hospital-mortality \
--fusion_type mmtm --layer_after 4 \
--labels_set mortality \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/ihm_ts.normalizer \
--save_dir checkpoints/mortality/mmtm_partial

# nohup sh scripts/mortality/train/mmtm.sh > ihm_mmtm_partial.log 2>&1 &