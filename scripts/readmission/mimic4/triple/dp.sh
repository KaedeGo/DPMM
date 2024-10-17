CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_3d.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--lr 0.0001 \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type dp \
--dp_fuse_type mha \
--dp 0.000001 \
--K 5 \
--rho_scale -3 \
--temperature 0.005 \
--save_dir checkpoints/readm/mimic4/triple/dp_mha
# nohup sh scripts/readmission/mimic4/triple/dp.sh > logs/readm_mimic4/triple/dp_mha.log 2>&1 &