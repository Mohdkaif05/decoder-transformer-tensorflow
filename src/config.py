# Model Configuration

VOCAB_SIZE = 8000
MAX_SEQ_LEN = 128

NUM_LAYERS = 4
D_MODEL = 256
NUM_HEADS = 4
DFF = 1024

DROPOUT_RATE = 0.1
TOP_K = 40

# Generation Configuration

MAX_NEW_TOKENS = 100
TEMPERATURE = 1.0

# File Paths

MODEL_WEIGHTS_PATH = "models/gpt_epoch_6.weights.h5"

TOKENIZER_MODEL_PATH = "tokenizer/tokenizer.model"
TOKENIZER_VOCAB_PATH = "tokenizer/tokenizer.vocab"