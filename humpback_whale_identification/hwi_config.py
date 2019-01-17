# Config file for Humpback whale identification.
from os import path

BASE_PATH = "/media3/kaggle/humpback_whale_identification/datasets"
TRAIN_IMAGES_PATH = path.sep.join([BASE_PATH, "train"])
TEST_IMAGES_PATH = path.sep.join([BASE_PATH, "test"])
GENERATED_TRAIN_IMAGES_PATH = path.sep.join([BASE_PATH, "train", "generated"])

# Mapping between training data and labels
IMAGE_ID_FILE = path.sep.join([BASE_PATH, "train.csv"])

# There are 5005 classes
NUM_CLASSES = 5005
NUM_TEST_IMAGES = 100000

# Output path
BASE_OUTPUT_PATH = "/media3/kaggle/humpback_whale_identification/output"
# TRAIN_LIST will contain the full image path and their corresponding label.
# Split the train list into two for train and validation
TRAIN_LIST = path.sep.join([BASE_OUTPUT_PATH, "list", "train.lst"])
VAL_LIST = path.sep.join([BASE_OUTPUT_PATH, "list", "val.lst"])
TEST_LIST = path.sep.join([BASE_OUTPUT_PATH, "list", "test.lst"])

# Encoded label and class name mapping file
ENC_CODE_CLASS_MAP = path.sep.join([BASE_OUTPUT_PATH, "enc_label_class_map.txt"])
# No of augmented images to be generated
GEN_IMAGES_FACTORS = [100, 50, 35, 25, 20, 18, 18, 15, 12, 10, 10, 9, 8, 8, 
                      8, 7, 7, 7, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
# No generation for the class having more than 100 samples as there is only one
NO_GEN_NUMBER = 100
# No of images to be generated for the class having 31 to 100 samples
MAX_GEN_FACTOR = 4
# Record file containing actual data with labels
TRAIN_REC = path.sep.join([BASE_OUTPUT_PATH, "record", "train.rec"])
VAL_REC = path.sep.join([BASE_OUTPUT_PATH, "record", "val.rec"])
TEST_REC = path.sep.join([BASE_OUTPUT_PATH, "record", "test.rec"])

# Dataset mean for "R", "G" and "B"
MEAN_PATH = path.sep.join([BASE_OUTPUT_PATH, "hwi_mean.json"]) 

BATCH_SIZE = 32
NUM_EPOCH = 100
# Number of available device
NUM_DEVICE = 1
