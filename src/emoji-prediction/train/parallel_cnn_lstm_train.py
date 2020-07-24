"""
Writen by Ehsan
"""
# pylint: disable-msg=no-member
# pylint: disable-msg=not-callable
import time
import torch
import torch.optim as optim
from torch import nn
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, f1_score
from model import CNNLSTM
from utils import DataSet
from utils import categorical_accuracy
from config import INPUT_PATH, GLOVE_PATH, LOG_PATH,\
    EMBEDDING_DIM, DROPOUT, N_FILTERS, FILTER_SIZE,\
    OUTPUT_SIZE, LSTM_UNITS, LSTM_LAYERS, BIDIRECTIONAL,\
    N_EPOCHS, MODEL_PATH, DEVICE, POS_EMBEDDING_DIM, TRANSFORM_LAYERS, SKIPGRAM_PATH, SENTI_EMBEDDING_5, SENTI_EMBEDDING_10


def count_parameters(input_model):
    """
    method for calculate number of model's parameter
    :param input_model: model
    :return: number of parameters
    """
    return sum(p.numel() for p in input_model.parameters() if p.requires_grad)


def epoch_time(s_time, e_time):
    """
    method for calculate time
    :param s_time: start time
    :param e_time: end time
    :return: Minutes and Seconds
    """
    elapsed_time = e_time - s_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


LOG_FILE = open(LOG_PATH, 'w')
DATA_SET = DataSet(INPUT_PATH, GLOVE_PATH, SKIPGRAM_PATH, SENTI_EMBEDDING_5)
DATA_SET.load_data()

MODEL = CNNLSTM(vocab_size=len(DATA_SET.vocab),  embedding_dim=EMBEDDING_DIM, pad_idx=DATA_SET.pad_idx,
                pos_tags_num=len(DATA_SET.pos_tags), pos_embedding_dim=POS_EMBEDDING_DIM, pos_pad_idx=DATA_SET.pos_pad_idx,
                n_filters=N_FILTERS, filter_sizes=FILTER_SIZE, lstm_units=LSTM_UNITS, lstm_layers=LSTM_LAYERS,
                bidirectional=BIDIRECTIONAL, dropout=DROPOUT, output_size=OUTPUT_SIZE, transorm_layers=TRANSFORM_LAYERS)

print(f'The model has {count_parameters(MODEL):,} trainable parameters')

MODEL.embeddings.weight.data.copy_(DATA_SET.embedding_vectors)
MODEL.embeddings.weight.data[DATA_SET.pad_idx] = torch.zeros(EMBEDDING_DIM)
MODEL.embeddings.weight.requires_grad = True

MODEL.pos_embeddings.weight.data[DATA_SET.pad_idx] = torch.zeros(POS_EMBEDDING_DIM)
torch.manual_seed(1234)
nn.init.uniform_(MODEL.pos_embeddings.weight, -1.0, 1.0)
MODEL.pos_embeddings.weight.requires_grad = True

'''
torch.manual_seed(1234)
nn.init.uniform_(model.embeddings.weight, -1.0, 1.0)
'''

OPTIMIZER = optim.Adam(MODEL.parameters())
CRITERION = nn.CrossEntropyLoss(weight=torch.tensor(DATA_SET.class_weight))

MODEL = MODEL.to(DEVICE)
CRITERION = CRITERION.to(DEVICE)


def train(model, iterator, optimizer, criterion):
    """
    method for train model
    :param model: your creation model
    :param iterator: train iterator
    :param optimizer: your optimizer
    :param criterion: your criterion
    :return: loss and accuracy for each epoch
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    iter_len = len(iterator)
    n_batch = 0

    for batch in iterator:
        n_batch += 1
        optimizer.zero_grad()

        predictions = model(batch.text, batch.pos)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if (n_batch % (iter_len//3)) == 0:
            print(f'\t train on: {(n_batch / iter_len) * 100:.2f}% of samples')
            print(f'\t accuracy : {(epoch_acc/n_batch)}')
            print(f'\t loss : {(epoch_loss/n_batch)}')
            print('________________________________________________\n')

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """
    method for evaluate model
    :param model: your creation model
    :param iterator: your iterator
    :param criterion: your criterion
    :return: loss, accuracy, precision, recall and F1 score for each epoch
    """
    evaluate_parameters_dict = {"loss": 0, "acc": 0, "precision": 0,
                                "recall": 0, "fscore": 0, "total_f1_score": 0}

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text, batch.pos)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)
            precision, recall, fscore, _ =\
                precision_recall_fscore_support(y_true=batch.label.cpu(),
                                                y_pred=np.argmax(predictions.cpu(),
                                                                 axis=1))
            _, _, total_f1_score, _ = \
                precision_recall_fscore_support(y_true=batch.label.cpu(),
                                                y_pred=np.argmax(predictions.cpu(), axis=1),
                                                average="weighted")

            """total_f1_score = fscore(y_true=batch.label.cpu(),
                                    y_pred=np.argmax(predictions.cpu(), axis=1),
                                    average='weighted')"""

            evaluate_parameters_dict["loss"] += loss.item()
            evaluate_parameters_dict["acc"] += acc.item()

            evaluate_parameters_dict["precision"] += precision
            evaluate_parameters_dict["recall"] += recall
            evaluate_parameters_dict["fscore"] += fscore
            evaluate_parameters_dict["total_f1_score"] += total_f1_score

    return evaluate_parameters_dict["loss"] / len(iterator),\
           evaluate_parameters_dict["acc"] / len(iterator),\
           evaluate_parameters_dict["precision"] / len(iterator),\
           evaluate_parameters_dict["recall"] / len(iterator),\
           evaluate_parameters_dict["fscore"] / len(iterator), \
           evaluate_parameters_dict["total_f1_score"] / len(iterator)


BEST_VALID_LOSS = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(MODEL, DATA_SET.iterator_list[0], OPTIMIZER, CRITERION)
    valid_loss, valid_acc, valid_precision, valid_recall, valid_fscore, total_f1_score =\
        evaluate(MODEL, DATA_SET.iterator_list[1], CRITERION)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < BEST_VALID_LOSS:
        BEST_VALID_LOSS = valid_loss
        torch.save(MODEL.state_dict(),
                   MODEL_PATH + 'model_epoch{}_loss{}.pt'.format(epoch+1, valid_loss))

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    print(f'\t Val. Precision: {valid_precision}')
    print(f'\t Val. Recall: {valid_recall}')
    print(f'\t Val. F1_Score: {valid_fscore}')
    print(f'\t Val. Total_F1_Score: {total_f1_score}')
    print('_____________________________________________________________________\n')

    LOG_FILE.write(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
    LOG_FILE.write(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%\n')
    LOG_FILE.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%\n')
    LOG_FILE.write(f'\t Val. Precision: {valid_precision}\n')
    LOG_FILE.write(f'\t Val. Recall: {valid_recall}\n')
    LOG_FILE.write(f'\t Val. F1_Score: {valid_fscore}\n')
    LOG_FILE.write(f'\t Val. Total_F1_Score: {total_f1_score}\n')
    LOG_FILE.write('____________________________________________________________\n')
    LOG_FILE.flush()
