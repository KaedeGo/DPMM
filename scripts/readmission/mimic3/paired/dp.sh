CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_benchmark_mimic3.py \
--dim 256 --dropout 0 --layers 2 \
--mode train \
--lr 0.0001 \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type dp \
--dp_fuse_type lstm \
--dp 0.00001 \
--save_dir checkpoints/readm/mimic3/paired/dp
# nohup sh scripts/readmission/mimic3/paired/copula.sh > logs/readm_mimic3/paired/dp.log 2>&1 &