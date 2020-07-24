"""
Writen by Ehsan
"""
# pylint: disable-msg=no-member
import torch

INPUT_PATH = "drive/My Drive/payanname/data/tweets_10_pos.csv"
GLOVE_PATH = "drive/My Drive/payanname/data/glove.6B.300d.txt"
SENTI_EMBEDDING_5 = "drive/My Drive/payanname/data/sswe5.6B.300d.txt"
SENTI_EMBEDDING_10 = "drive/My Drive/payanname/data/sswe10_2.6B.300d.txt"
SKIPGRAM_PATH = "drive/My Drive/payanname/data/skipgram_tweets_10.txt"
RESORCES_PATH = "drive/My Drive/payanname/resources/postagger.model"
LOG_PATH = 'log_all_10.txt'
MODEL_PATH = 'drive/My Drive/payanname/codes/model_2/model'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
EMBEDDING_DIM = 900
POS_EMBEDDING_DIM = 20
DROPOUT = 0.3
N_FILTERS = 128
FILTER_SIZE = [3, 4, 5]
OUTPUT_SIZE = 10
N_EPOCHS = 20
LSTM_UNITS = 128
LSTM_LAYERS = 1
TRANSFORM_LAYERS = 3
BIDIRECTIONAL = True
