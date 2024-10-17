CUDA_VISIBLE_DEVICES=6 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 1e-04 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type dp \
--dp 1 \
--temperature 0.005 \
--replace_w_align na \
--dp_fuse_type mha \
--save_dir checkpoints/readm/mimic4/abalation/copula_paired_w_na
# nohup sh scripts/readmission/mimic4/abalation/copula_paired_w_align.sh > logs/readm_mimic4/abalation/copula_paired_w_na.log 2>&1 &