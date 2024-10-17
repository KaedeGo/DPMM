CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python fusion_main_3d.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--lr 0.0001 \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task in-hospital-mortality \
--labels_set mortality \
--fusion_type dp \
--dp_fuse_type mha \
--dp 0.000001 \
--K 2 \
--rho_scale -2.5 \
--temperature 0.005 \
--save_dir checkpoints/mortality/mimic4/triple/dp_mha
# nohup sh scripts/mortality/mimic4/triple/dp.sh > logs/ihm_mimic4/triple/dp_mha.log 2>&1 &