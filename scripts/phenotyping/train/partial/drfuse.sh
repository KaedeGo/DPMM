CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1 python fusion_main.py \
--dim 256 --dropout 0.3 --layers 2 \
--mode train \
--epochs 200 --batch_size 32 --lr 0.0001 \
--vision_num_classes 25 --num_classes 25 \
--data_pairs partial_ehr_cxr \
--fusion_type drfuse --layer_after 4 \
--task phenotyping_48h \
--labels_set pheno \
--normalizer_state /disk1/fwu/myProjects/MedFuse/data/pheno_ts.normalizer \
--save_dir checkpoints/phenotyping/partial/drfuse

# nohup sh scripts/phenotyping/train/partial/drfuse.sh > logs/phe_mimic4/partial/drfuse.log 2>&1 &
