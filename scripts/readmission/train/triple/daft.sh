CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 python fusion_main_3d.py \
--dim 256 --dropout 0.3 --layers 2 \
--vision_backbone resnet34 \
--mode train \
--epochs 100 --batch_size 16 --lr 6.493e-05 \
--vision_num_classes 1 --num_classes 1 \
--data_pairs paired_ehr_cxr \
--fusion_type daft --layer_after 4 \
--labels_set readm \
--task readmission \
--normalizer_state /data1/fwu/myProjects/MedFuse/data/readm_ts.normalizer \
--save_dir checkpoints/readm/triple/daft

# nohup sh scripts/readmission/train/triple/daft.sh > logs/readm_mimic4/triple/daft.log 2>&1 &