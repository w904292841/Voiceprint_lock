
# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512 # FFT（快速傅立叶变换）是指通过在计算项中使用对称性可以有效地计算离散傅立叶变换（DFT）的方法。当n为2的幂时，对称性最高，因此，对于这些大小，变换效率最高。
BUCKET_STEP = 1
MAX_SEC = 10
DIM = (1, 512, 300)
MFCC_DIM = (3, 300, 16)

# Model
WEIGHTS_FILE = "weights/weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,300,1)

# IO on windows
FA_DIR = "H:/科研/mfcc_data_pre/"
TRAIN_LIST_FILE = "../a_iden/new_train.csv"
PERSONAL_WEIGHT = "../weights/weight_128.h5"

# IO on linux
# FA_DIR = "/home/longfuhui/all_data/vox1-dev-wav/wav/"
# PERSONAL_WEIGHT = "weights/weight_128.h5"
# RESULT_FILE = "results/results.csv"

# train
TENSORBOARD_LOG_PATH = "tensorboard/log"
LOSS_PNG = "img/loss.png"
ACC_PNG = "img/acc.png"
CONTINUE_TRAINING = 0
SAVE = 1
LR = 1e-3
VERI_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 128
WEIGHT_DECAY = 5e-5
TEST_BATCH_SIZE = 1
IDEN_CLASS = 1024
N_CLASS = 1251
OUT_CHANNEL = 512
IDEN_NUMBER = 898
VERI_NUMBER = 10
TRAIN_NUMBER = 14793
MARGIN = 0.35

ALL_BATCH_SIZE = 2
ALL_EPOCHS = 2
VERI_BATCH_SIZE = 128
'''
Identification
'''
# train
IDEN_TRAIN_LIST_FILE = "../a_iden/train_list_pt_noi_red.txt"
IDEN_MODEL_FA_PATH = "../pytorch_models/iden/m_128"
IDEN_MODEL_PATH = "../pytorch_models/iden/m_1024/mfcc_iden_model_1024.bin"

# test
IDEN_TEST_FILE = "../a_iden/test_list_pt_noi_red.txt"
IDEN_MODEL_LOAD_PATH = "../models/iden/iden_model_1024.h5"

'''
verification
'''
# train
VERI_TRAIN_LIST_FILE = "../a_veri/train_veri_list_pt_1024.txt"

# test
VERI_TEST_LIST_FILE = "../a_veri/test_veri_list_pt.txt"
VERI_DEV_LIST_FILE = "../a_veri/test_veri_list_pt_1024.txt"
VERI_MODEL_LOAD_PATH = "../pytorch_models/veri/m_1024/veri_model_1024"

# train
MODEL_FA_PATH = "../pytorch_models/wav_dev/"
MODEL_PATH = "../pytorch_models/wav_dev/m_1251.bin"

VERI_MODEL_FA_PATH = "../pytorch_models/veri/m_1024/"
VERI_MODEL_PATH = "../pytorch_models/veri/m_1024/veri_model_1024.bin"
SYMBOL_PATH = "../pytorch_models/veri/m_128/symbol.pt"
SYMBOL_TEST_PATH = "../pytorch_models/veri/m_128/symbol_test.pt"

WEIGHT_PATH = "../pytorch_models/iden/m_1024/checkpoint/autoencoder.t7"

# mfcc
MFCC_IDEN_MODEL_PATH = "../pytorch_models/iden/m_1024/mfcc_iden_model_1024.bin"
