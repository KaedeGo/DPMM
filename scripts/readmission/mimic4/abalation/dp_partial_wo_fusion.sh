CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic4.py \
--dim 256 --dropout 0 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 32 --lr 1e-04 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type dp \
--dp 0.000001 \
--K 3 \
--rho_scale -2.5 \
--temperature 0.005 \
--dp_resample True \
--dp_fuse_type na \
--save_dir checkpoints/readm/mimic4/abalation/mimic4/dp_partial_wo_fusion
# nohup sh scripts/readmission/mimic4/abalation/dp_partial_wo_fusion.sh > logs/readm_mimic4/abalation/dp_partial_wo_fusion.log 2>&1 &