CUDA_VISIBLE_DEVICES=3 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0.3 --layers 2 \
--lr 1e-04 \
--vision_backbone resnet34 \
--copula 0 \
--align 0 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type uni_ehr \
--save_dir checkpoints/mortality/mimic4/paired/ehr_only

# nohup sh scripts/mortality/mimic4/paired/ehr_only.sh > logs/ihm_mimic4/paired/ehr_only.log 2>&1 &