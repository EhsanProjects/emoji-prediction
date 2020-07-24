"""
Written by Ehsan
"""
# pylint: disable-msg=no-member
import random
import hazm
import torch
import numpy as np
import pickle as pkl
import pandas as pd
from torchtext import data
from torchtext.vocab import Vectors
from sklearn.utils import class_weight
from config import DEVICE, BATCH_SIZE, RESORCES_PATH


class DataSet:
    """
    DataSet Class use for preparing data
    and iterator for training model.
    """
    def __init__(self, input_data_path, glove_path, skipg_path, sentiem_path):
        self.files_address = [input_data_path, glove_path, skipg_path, sentiem_path]
        self.glove_path = glove_path
        self.skipg_path = skipg_path
        self.sentiem_path = sentiem_path
        self.tagger = hazm.POSTagger(model=RESORCES_PATH)
        self.iterator_list = []
        self.class_weight = None
        self.embedding_vectors = None
        self.pad_idx = None
        self.unk_idx = None
        self.vocab = None

    def read_csv_file(self):
        """
        Reading input csv file and calculate class_weight
        :return: dataFrame
        """
        input_df = pd.read_csv(self.files_address[0])
        input_df = input_df.astype({'tweets': 'str'})
        input_df = input_df.astype({'emojis': 'str'})
        # self.class_weight = class_weight.compute_class_weight('balanced',
        #                                                       np.unique(input_df.emojis),
        #                                                       input_df.emojis)
        return input_df

    def pos_tagger(self, input_sen):
        output = ""
        pos = self.tagger.tag(hazm.word_tokenize(input_sen))
        for word in pos:
            output = output + " " + str(word[1])
        return output.strip().split()

    def load_data(self):
        """
        Create iterator for train and test data
        """
        # Create Field for data
        text_field = data.Field(tokenize=hazm.word_tokenize, batch_first=True)
        pos_field = data.Field(tokenize=self.pos_tagger, batch_first=True)
        label_field = data.LabelField()
        datafields = [("text", text_field), ("pos", pos_field), ("label", label_field)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        all_examples = [data.Example.fromlist(i, datafields) for i in
                        self.read_csv_file().values.tolist()]
        all_data = data.Dataset(all_examples, datafields)

        seed = 1234
        train_data, test_data = all_data.split(split_ratio=0.85, random_state=random.seed(seed))

        text_field.build_vocab(train_data,
                               min_freq=10,
                               unk_init=torch.Tensor.normal_,
                               vectors=[Vectors(self.skipg_path), Vectors(self.glove_path), Vectors(self.sentiem_path)])

        pos_field.build_vocab(train_data)
        self.pos_tags = pos_field.vocab
        self.pos_pad_idx = pos_field.vocab.stoi[pos_field.pad_token]
        self.pos_unk_idx = pos_field.vocab.stoi[pos_field.unk_token]

        self.embedding_vectors = text_field.vocab.vectors
        self.vocab = text_field.vocab
        self.pad_idx = text_field.vocab.stoi[text_field.pad_token]
        self.unk_idx = text_field.vocab.stoi[text_field.unk_token]

        label_field.build_vocab(train_data)
        label_list = []
        for emoji, idx in label_field.vocab.stoi.items():
            for num in range(label_field.vocab.freqs[emoji]):
                label_list.append(idx)

        data_class_weight = class_weight.compute_class_weight('balanced',
                                                              np.unique(label_list),
                                                              label_list)
        self.class_weight = data_class_weight.astype(np.float32)

        with open("tag.pkl", "wb") as pkl_data:
            pkl.dump(label_field.vocab.stoi, pkl_data)

        train_iterator = data.BucketIterator(
            train_data,
            batch_size=BATCH_SIZE,
            sort=False,
            device=DEVICE)

        test_iterator = data.BucketIterator(
            test_data,
            batch_size=1024,
            sort=False,
            device=DEVICE)

        self.iterator_list = [train_iterator, test_iterator]

        print("Loaded {} training examples".format(len(train_data)))
        print("Loaded {} test examples".format(len(test_data)))


def binary_accuracy(preds, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == target).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def categorical_accuracy(preds, target):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)# get the index of the max probability
    correct = max_preds.squeeze(1).eq(target)
    return correct.sum() / torch.FloatTensor([target.shape[0]])

"""from config import *
mc = DataSet(INPUT_PATH, GLOVE_PATH)
mc.load_data()"""
