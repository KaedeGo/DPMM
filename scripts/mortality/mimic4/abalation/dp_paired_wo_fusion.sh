CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 1e-04 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type dp \
--dp 0.000001 \
--K 2 \
--rho_scale -2.5 \
--dp_fuse_type na \
--temperature 0.005 \
--save_dir checkpoints/mortality/mimic4/abalation/dp_paired_wo_fusion
# nohup sh scripts/mortality/mimic4/abalation/dp_paired_wo_fusion.sh > logs/ihm_mimic4/abalation/dp_paired_wo_fusion.log 2>&1 &