from pathlib import Path

ROOT_PATH = Path(__file__).parent

MIMIC4_DATA_DIR = "/disk1/fwu/myProjects/MedFuse/data_mimic4/"

# MIMIC3_DATA_DIR = "/disk1/fwu/myProjects/MedFuse/data_mimic3/"
MIMIC3_DATA_DIR = "/home/fwu/Datasets/myProjects/MultimodalMIMIC/data_mimic3/"

CXR_DATA_DIR = "/disk1/fwu/myProjects/MedFuse/data_mimic4/mimic-cxr/"

# treat normalizer_readm as the normalizer for both readmission and in-hospital-mortality
MIMIC4_NORMALIZER_PATH = "/disk1/fwu/myProjects/MedFuse/data_mimic4/readm_ts.normalizer"

# MIMIC3_NORMALIZER_PATH = "/disk1/fwu/myProjects/MedFuse/data_mimic3/readm_ts.normalizer"
MIMIC3_NORMALIZER_PATH = "/home/fwu/Datasets/myProjects/MultimodalMIMIC/data_mimic3/readm_ts.normalizer"