import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

CLASSES = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]


class MIMIC_NOTE_EHR(Dataset):
    def __init__(self, args, ehr_ds, note_ds, split="train"):

        self.CLASSES = CLASSES
        self.files_paired = note_ds.filtered_names
        self.files_all = ehr_ds.names
        self.files_unpaired = sorted(list(set(self.files_all) - set(self.files_paired)))
        self.ehr_ds = ehr_ds
        self.note_ds = note_ds
        self.args = args
        self.split = split
        self.data_ratio = self.args.data_ratio
        if split == "test":
            self.data_ratio = 1.0
        print(
            f"split: {split}, ehr_files_all: {len(self.files_all)}, ehr_files_paired: {len(self.files_paired)}"
        )

    def __getitem__(self, index):
        if self.args.data_pairs == "paired_ehr_note":
            ehr_data, labels_ehr = self.ehr_ds[self.files_paired[index]]
            text_token, atten_mask, labels_note = self.note_ds[self.files_paired[index]]
            return ehr_data, text_token, atten_mask, labels_ehr, labels_note
        elif self.args.data_pairs == "paired_ehr":
            ehr_data, labels_ehr = self.ehr_ds[self.files_paired[index]]
            text_token, atten_mask, labels_note = None, None, None
            return ehr_data, text_token, atten_mask, labels_ehr, labels_note
        elif self.args.data_pairs == "note":
            ehr_data, labels_ehr = np.zeros((1, 10)), np.zeros(self.args.num_classes)
            text_token, atten_mask, labels_note = self.note_ds[self.files_paired[index]]
            return ehr_data, text_token, atten_mask, labels_ehr, labels_note
        elif self.args.data_pairs == "partial_ehr":
            ehr_data, labels_ehr = self.ehr_ds[self.files_all[index]]
            text_token, atten_mask, labels_note = None, None, None
            return ehr_data, text_token, atten_mask, labels_ehr, labels_note

        elif self.args.data_pairs == "partial_ehr_note":
            if index < len(self.files_paired):
                ehr_data, labels_ehr = self.ehr_ds[self.files_paired[index]]
                text_token, atten_mask, labels_note = self.note_ds[
                    self.files_paired[index]
                ]
            else:
                ehr_data, labels_ehr = self.ehr_ds[
                    self.files_unpaired[index - len(self.files_paired)]
                ]
                text_token, atten_mask, labels_note = None, None, None
            return ehr_data, text_token, atten_mask, labels_ehr, labels_note

    def __len__(self):
        if "paired" in self.args.data_pairs:
            return len(self.files_paired)
        elif self.args.data_pairs == "partial_ehr":
            return len(self.files_all)
        elif self.args.data_pairs == "note":
            return len(self.files_paired)
        elif self.args.data_pairs == "partial_ehr_note":
            return len(self.files_paired) + int(
                self.data_ratio * len(self.files_unpaired)
            )


def load_note_ehr(
    args,
    ehr_train_ds,
    ehr_val_ds,
    note_train_ds,
    note_val_ds,
    ehr_test_ds,
    note_test_ds,
):

    train_ds = MIMIC_NOTE_EHR(args, ehr_train_ds, note_train_ds)
    val_ds = MIMIC_NOTE_EHR(args, ehr_val_ds, note_val_ds, split="val")
    test_ds = MIMIC_NOTE_EHR(args, ehr_test_ds, note_test_ds, split="test")

    train_dl = DataLoader(
        train_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=my_collate,
        pin_memory=True,
        num_workers=16,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        args.batch_size,
        shuffle=False,
        collate_fn=my_collate,
        pin_memory=True,
        num_workers=16,
        drop_last=False,
    )
    test_dl = DataLoader(
        test_ds,
        args.batch_size,
        shuffle=False,
        collate_fn=my_collate,
        pin_memory=True,
        num_workers=16,
        drop_last=False,
    )

    return train_dl, val_dl, test_dl


def printPrevalence(merged_file, args):
    if args.labels_set == "pheno":
        total_rows = len(merged_file)
        print(merged_file[CLASSES].sum() / total_rows)
    else:
        total_rows = len(merged_file)
        print(merged_file["y_true"].value_counts())


def my_collate(batch):
    x = [item[0] for item in batch]
    pairs = [False if item[1] is None else True for item in batch]
    token = torch.stack(
        [
            torch.zeros((5, 512), dtype=torch.int64) if item[1] is None else item[1]
            for item in batch
        ]
    )
    mask = torch.stack(
        [
            torch.zeros((5, 512), dtype=torch.int64) if item[2] is None else item[2]
            for item in batch
        ]
    )
    x, seq_length = pad_zeros(x)
    targets_ehr = np.array([item[3] for item in batch])
    targets_note = torch.stack(
        [torch.tensor(0.0) if item[4] is None else item[4] for item in batch]
    )
    return [x, token, mask, targets_ehr, targets_note, seq_length, pairs]


def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_length = [x.shape[0] for x in arr]
    max_len = max(seq_length)
    ret = [
        np.concatenate(
            [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0
        )
        for x in arr
    ]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [
            np.concatenate(
                [x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)],
                axis=0,
            )
            for x in ret
        ]
    return np.array(ret), seq_length


if __name__ == "__main__":
    import sys

    sys.path.append("/home/fwu/Documents/myProjects/MedFuse/")
    from arguments import args_parser
    from ehr_utils.preprocessing import Discretizer, Normalizer
    from dataset_mimic3.ehr_dataset import get_datasets
    from dataset_mimic3.note_dataset import get_note_datasets

    parser = args_parser()
    args = parser.parse_args()
    args.ehr_data_dir = "/disk1/fwu/myProjects/MedFuse/data_mimic3"
    args.data_pairs = "partial_ehr_note"
    args.labels_set = "mortality"
    args.task = "readmission"
    args.normalizer_state = (
        "/disk1/fwu/myProjects/MedFuse/data_mimic3/ihm_ts.normalizer"
    )

    def read_timeseries(args):
        path = f"{args.ehr_data_dir}/{args.task}/train/3_episode1_timeseries.csv"
        ret = []
        with open(path, "r") as tsfile:
            header = tsfile.readline().strip().split(",")
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(",")
                ret.append(np.array(mas))
        return np.stack(ret)

    discretizer = Discretizer(
        timestep=float(args.timestep),
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
    )

    discretizer_header = discretizer.transform(read_timeseries(args))[1].split(",")
    cont_channels = [
        i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1
    ]

    normalizer = Normalizer(
        fields=cont_channels
    )  # choose here which columns to standardize
    normalizer_state = args.normalizer_state

    normalizer.load_params(normalizer_state)

    ehr_train_ds, ehr_val_ds, ehr_test_ds = get_datasets(discretizer, normalizer, args)

    note_train_ds, note_val_ds, note_test_ds = get_note_datasets(args)

    train_dl, val_dl, test_dl = load_note_ehr(
        args,
        ehr_train_ds,
        ehr_val_ds,
        note_train_ds,
        note_val_ds,
        ehr_test_ds,
        note_test_ds,
    )
    pos_train = 0
    pos_val = 0
    pos_test = 0
    for i, (x, token, mask, y_ehr, y_note, seq_lengths, pairs) in enumerate(train_dl):
        pos_train += y_ehr.sum(axis=0)
    for i, (x, token, mask, y_ehr, y_note, seq_lengths, pairs) in enumerate(val_dl):
        pos_val += y_ehr.sum(axis=0)
    for i, (x, token, mask, y_ehr, y_note, seq_lengths, pairs) in enumerate(test_dl):
        pos_test += y_ehr.sum(axis=0)
    print(
        f"Task: {args.task},\tData Pairs: {args.data_pairs}, \nPositives: Train: {pos_train}, Val: {pos_val}, Test: {pos_test}, Total: {pos_train+pos_val+pos_test}"
    )
"""
in-hospital-mortality:
    split: train, ehr_files_all: 14681, ehr_files_paired: 3652
    split: val, ehr_files_all: 3222, ehr_files_paired: 815
    split: test, ehr_files_all: 3236, ehr_files_paired: 806
    total: unmatched: 21139, matched: 5273

    Data Pairs: partial_ehr_note, Positives: Train: 1985, Val: 436, Test: 374, Total: 2795
    Data Pairs: paired_ehr_note, Positives: Train: 512, Val: 125, Test: 99, Total: 736

readmission:
    split: train, ehr_files_all: 14681, ehr_files_paired: 3652
    split: val, ehr_files_all: 3222, ehr_files_paired: 815
    split: test, ehr_files_all: 3236, ehr_files_paired: 806
    total: unmatched: 21139, matched: 5273

    Data Pairs: paired_ehr_note, Positives: Train: 692, Val: 167, Test: 139, Total: 998

phenotyping_48h:
    split: train, ehr_files_all: 14681, ehr_files_paired: 3652
    split: val, ehr_files_all: 3222, ehr_files_paired: 815
    split: test, ehr_files_all: 3236, ehr_files_paired: 806
    total: unmatched: 21139, matched: 5273

"""
