CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0 --layers 2 \
--mode train \
--lr 0.0001 \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs partial_ehr_note \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.000001 \
--copula_family Frank \
--save_dir checkpoints/readm/mimic3/partial/copula
# nohup sh scripts/readmission/mimic3/partial/copula.sh > logs/readm_mimic3/partial/copula.log 2>&1 &