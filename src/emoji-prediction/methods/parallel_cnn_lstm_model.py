"""
Written by Ehsan
"""
# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ
import torch
from torch import nn
import torch.nn.functional as F
from utils import DataSet
from config import *
import numpy as np


class CNNLSTM(nn.Module):
    """
    Class for CNN_LSTM Model
    """
    def __init__(self, **kwargs):
        """parameters_list[vocab_size, embedding_dim, pad_idx, n_filters, filter_sizes,
        lstm_units, lstm_layers, bidirectional, dropout, output_size]"""
        super().__init__()

        self.cnn_filters = kwargs["n_filters"]
        self.cnn_filter_size = kwargs["filter_sizes"]

        self.embeddings = nn.Embedding(kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])
        self.embeddings.weight.requires_grad = True

        self.pos_embeddings = nn.Embedding(kwargs["pos_tags_num"],
                                           embedding_dim=kwargs["pos_embedding_dim"],
                                           padding_idx=kwargs["pos_pad_idx"])
        self.pos_embeddings.weight.requires_grad = True

        self.rnn = nn.LSTM(input_size=kwargs["embedding_dim"] + kwargs["pos_embedding_dim"],
                           hidden_size=kwargs["lstm_units"],
                           num_layers=kwargs["lstm_layers"],
                           bidirectional=kwargs["bidirectional"],
                           dropout=kwargs["dropout"])

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=3,
                      out_channels=kwargs["n_filters"],
                      kernel_size=(fs, kwargs["lstm_units"]//2))
            for fs in kwargs["filter_sizes"]
        ])

        self.fully_connected = nn.Linear((len(kwargs["filter_sizes"]) * kwargs["n_filters"]) +
                                         (kwargs["lstm_units"] * 2),
                                         kwargs["output_size"])

        self.fc_layer = nn.Linear(in_features=384, out_features=128)
        self.output_layer = nn.Linear(in_features=128, out_features=OUTPUT_SIZE)

        self.transform = nn.ModuleList([
                            nn.Linear(in_features=kwargs["lstm_units"] * 2,
                                      out_features=kwargs["lstm_units"]//2)
                            for fs in range(kwargs["transorm_layers"])
        ])

        self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self, input_text, input_pos):
        # text = [batch size, sent len]
        word_embedded = self.embeddings(input_text)
        pos_embedded = self.pos_embeddings(input_pos)
        embedded = torch.cat((word_embedded, pos_embedded), dim=2)
        embedded = self.dropout(embedded)
        # embedded = [batch size, sent len, emb dim]

        embedded_lstm = embedded.permute(1, 0, 2)
        # embedded_lstm = [sent len, batch size, emb dim]

        output, (hidden, _) = self.rnn(embedded_lstm)
        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # _ = [num layers * num directions, batch size, hid dim]
        output = self.dropout(output)
        output = F.relu(output)

        output = output.permute(1, 0, 2)
        # output = [batch size, sent len, hid dim * num directions]
        transformed = [self.dropout(F.relu(trans(output)))for trans in self.transform]

        stack_transform = torch.stack(transformed)
        stack_transform = stack_transform.permute(1, 0, 2, 3)

        conved = [F.relu(conv(stack_transform)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat_cnn = torch.cat(pooled, dim=1)
        cat_cnn = self.dropout(cat_cnn)
        # cat_cnn = [batch size, n_filters * len(filter_sizes)]

        fc = F.relu(self.fc_layer(cat_cnn))

        return self.output_layer(fc)


"""model = CNNLSTM(vocab_size=len(dataset.vocab),  embedding_dim=EMBEDDING_DIM, pad_idx=dataset.pad_idx,
                pos_tags_num=len(dataset.pos_tags), pos_embedding_dim=POS_EMBEDDING_DIM, pos_pad_idx=dataset.pos_pad_idx,
                n_filters=N_FILTERS, filter_sizes=FILTER_SIZE, lstm_units=LSTM_UNITS, lstm_layers=LSTM_LAYERS,
                bidirectional=BIDIRECTIONAL, dropout=DROPOUT, output_size=OUTPUT_SIZE, transorm_layers=TRANSFORM_LAYERS)

text = torch.rand((BATCH_SIZE, 100))

print(model.forward(text.long(), text.long()).shape)"""
