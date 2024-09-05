CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--lr 9.382e-05 \
--vision_backbone resnet34 \
--copula 0 \
--mode train \
--epochs 100 --batch_size 32 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--data_ratio 1.0 \
--task readmission \
--labels_set readm \
--fusion_type lstm \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/readm_ts.normalizer \
--save_dir checkpoints/readm/paired/medFuse
# nohup sh scripts/readmission/train/paired/medFuse.sh > logs/readm_mimic4/paired/medFuse.log 2>&1 &