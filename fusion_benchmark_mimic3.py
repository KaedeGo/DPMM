from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import random

from trainers.mimic3.fusion_trainer import FusionTrainer
from trainers.mimic3.mmtm_trainer import MMTMTrainer
from trainers.mimic3.daft_trainer import DAFTTrainer
from trainers.mimic3.drfuse_trainer import DrFuseTrainer
from trainers.mimic3.copula_trainer import CopulaTrainer
from trainers.mimic3.dp_trainer import DirichletProcessTrainer

from ehr_utils.preprocessing import Discretizer, Normalizer
from dataset_mimic3.ehr_dataset import get_datasets
from dataset_mimic3.note_dataset import get_note_datasets
from dataset_mimic3.fusion import load_note_ehr
from pathlib import Path
from paths import *
import torch
from arguments import args_parser

import wandb

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"set seed {seed}")

def read_timeseries(args):
    path = f'{args.ehr_data_dir}/{args.task}/train/3_episode1_timeseries.csv'
    ret = []
    with open(path, "r") as tsfile:
        header = tsfile.readline().strip().split(',')
        assert header[0] == "Hours"
        for line in tsfile:
            mas = line.strip().split(',')
            ret.append(np.array(mas))
    return np.stack(ret)

def get_ehr_dataset_info(args):
    discretizer = Discretizer(timestep=float(args.timestep),
                            store_masks=True,
                            impute_strategy='previous',
                            start_time='zero')

    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = args.normalizer_state
    if normalizer_state is None:
        normalizer_state = 'normalizers/ph_ts{}.input_str:previous.start_time:zero.normalizer'.format(args.timestep)
        normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
    normalizer.load_params(normalizer_state)
    return discretizer,normalizer

if __name__ == "__main__":
    parser = args_parser()
    # add more arguments here ...
    args = parser.parse_args()

    args.ehr_data_dir = MIMIC3_DATA_DIR
    args.cxr_data_dir = CXR_DATA_DIR
    args.normalizer_state = MIMIC3_NORMALIZER_PATH

    # args.batch_size = 8
    # args.ehr_data_dir = '/disk1/fwu/myProjects/MedFuse/data_mimic3/'
    # args.data_pairs = "paired_ehr_note"

    print(args)
    
    # create a directory to save the results
    path = Path(args.save_dir)
    path.mkdir(parents=True, exist_ok=True)

    # set seed
    seed_torch(1002)

    discretizer, normalizer = get_ehr_dataset_info(args)

    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)

    note_train_ds, note_val_ds, note_test_ds = get_note_datasets(args)

    train_dl, val_dl, test_dl = load_note_ehr(args, ehr_train_ds, ehr_val_ds, note_train_ds, note_val_ds, ehr_test_ds, note_test_ds)

    for lr in [0.0001, 0.0005, 0.001]:
    # for dr in [0, 0.05, 0.1, 0.2, 0.3, 0.4]:
        for temperature in [0.001, 0.005, 0.01]:
            for rho_scale in [-2.5, -3, -3.5, -4]:
                for K in [2, 3]:
                    if "partial" in args.data_pairs:
                        pair_type = "partial"
                    else:
                        pair_type = "paired"

                    if "mimic3" in args.ehr_data_dir:
                        dataset_name = "mimic3"
                    else:
                        dataset_name = "mimic4"

                    # args.lr = lr
                    # args.dropout = dr
                    args.K = K
                    args.rho_scale = rho_scale
                    args.temperature = temperature

                    name = f"{dataset_name}_{args.fusion_type}_{pair_type}_{args.labels_set}_rho{rho_scale}_K{K}_temp{args.temperature}"
                    # name = f"{args.fusion_type}_{pair_type}_{args.labels_set}_dr{dr}"
                    # name = f"{args.fusion_type}_{pair_type}_{args.labels_set}_lr{lr}"
                    # name = f"{args.fusion_type}_{pair_type}_{args.labels_set}_temp{args.temperature}"

                    args.save_dir = f"checkpoints/phenotyping/paired/dp/{name}"
                    path = Path(args.save_dir)
                    path.mkdir(parents=True, exist_ok=True)

                    config = vars(args)
                    mode = "online"

                    wandb.init(name=name,
                               project='DPMMM',
                               notes="",
                               mode=mode,
                               config=config,
                               tags=["dp", pair_type]
                               )

                    with open(f"{args.save_dir}/args.txt", 'w') as results_file:
                        for arg in vars(args):
                            print(f"  {arg:<40}: {getattr(args, arg)}")
                            results_file.write(f"  {arg:<40}: {getattr(args, arg)}\n")

                    if args.fusion_type == 'mmtm':
                        trainer = MMTMTrainer(
                            train_dl,
                            val_dl,
                            args,
                            test_dl=test_dl
                        )
                    elif args.fusion_type == 'daft':
                        trainer = DAFTTrainer(train_dl,
                                              val_dl,
                                              args,
                                              test_dl=test_dl)
                    elif args.fusion_type == 'drfuse':
                        trainer = DrFuseTrainer(
                            train_dl,
                            val_dl,
                            args,
                            test_dl=test_dl
                        )
                    elif args.fusion_type == 'copula':
                        trainer = CopulaTrainer(
                            train_dl,
                            val_dl,
                            args,
                            test_dl=test_dl
                        )
                    elif args.fusion_type == 'dp':
                        trainer = DirichletProcessTrainer(
                            train_dl,
                            val_dl,
                            args,
                            test_dl=test_dl
                        )
                    else:
                        trainer = FusionTrainer(
                            train_dl,
                            val_dl,
                            args,
                            test_dl=test_dl
                        )

                    if args.mode == 'train':
                        print("==> training")

                        trainer.train()
                        trainer.args.load_state = args.save_dir + '/best_checkpoint.pth.tar'
                        trainer.load_state()
                        trainer.eval()

                        wandb.finish()
                        args.load_state = None

                    elif args.mode == 'eval':
                        trainer.eval()
                    else:
                        raise ValueError("not Implementation for args.mode")