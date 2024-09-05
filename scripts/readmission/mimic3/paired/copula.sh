CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main_mimic3.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 100 --batch_size 16 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_note \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type copula \
--copula_fuse_type lstm \
--copula 0.00005 \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data_mimic3/readm_ts.normalizer \
--ehr_data_dir /disk1/fwu/myProjects/MedFuse/data_mimic3 \
--save_dir checkpoints/readm/mimic3/paired/copula
# nohup sh scripts/readmission/mimic3/paired/copula.sh > logs/readm_mimic4/paired/copula.log 2>&1 &