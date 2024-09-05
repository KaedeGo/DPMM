CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 16 --lr 4.355e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--task in-hospital-mortality \
--fusion_type mmtm --layer_after 4 \
--labels_set mortality \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data_mimic3/ihm_ts.normalizer \
--ehr_data_dir /disk1/fwu/myProjects/MedFuse/data_mimic3 \
--save_dir checkpoints/mortality/mimic3/paired/mmtm
# nohup sh scripts/mortality/mimic3/paired/mmtm.sh > logs/ihm_mimic3/paired/mmtm.log 2>&1 &