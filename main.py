import os
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>',
                 sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1

        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt files and generate parallel text dataset:
# self.data[idx][0] is the tensor of source sequence
# self.data[idx][1] is the tensor of target sequence
# See examples in the cell below.
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, tgt_file_path, src_vocab=None,
                 tgt_vocab=None, extend_vocab=False, device='cuda'):
        (self.data, self.src_vocab, self.tgt_vocab, self.src_max_seq_length,
         self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, tgt_file_path, src_vocab, tgt_vocab, extend_vocab,
            device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None,
                              tgt_vocab=None, extend_vocab=False,
                              device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()

        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]  # remove line break
                length = len(tokens)
                if src_max < length:
                    src_max = length
        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens
        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                for token in tokens:
                    seq.append(src_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                # append a start token
                seq.append(tgt_sos_idx)
                for token in tokens:
                    seq.append(tgt_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                # append an end token
                seq.append(tgt_eos_idx)

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")

        return data_list, src_vocab, tgt_vocab, src_max, tgt_max

# `DATASET_DIR` should be modified to the directory where you downloaded
# the dataset. On Colab, use any method you like to access the data
# e.g. upload directly or access from Drive, ...

DATASET_DIR = "./"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

TASK = "numbers__place_value"
# TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True, device=DEVICE)

# get the vocab
src_vocab = train_set.src_vocab
tgt_vocab = train_set.tgt_vocab

src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
tgt_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, tgt_file_path, src_vocab=src_vocab, tgt_vocab=tgt_vocab,
    extend_vocab=False, device=DEVICE)

from torch.utils.data import DataLoader

batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False)

########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch_size, sequence length, embed dim]
        """
        assert x.shape[1] < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")

        x = x + self.pe[:, :x.shape[1]]

        return self.dropout(x)

class Model(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Model, self).__init__()
        self.emb_size = emb_size

        # embedding
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size)

        #transformer
        self.transformer = nn.Transformer(d_model=emb_size,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)

        #positional encoder
        self.pos_enc = PositionalEncoding(emb_size)

        #classifier
        self.linear = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt):

        #mask creation
        src_mask, src_padding_mask = self.create_mask_src(src)
        tgt_mask, tgt_padding_mask = self.create_mask_tgt(tgt)

        #embedding + encoding
        src_emb = self.pos_enc(self.src_embedding(src))
        tgt_emb = self.pos_enc(self.tgt_embedding(tgt))

        #transformer
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)

        #classifier
        return self.linear(outs)

    def encode(self, src, mask, src_padding_mask):
        #embedding + positional encoding
        emb = self.pos_enc(self.src_embedding(src))

        #return the memory of the encoder
        return self.transformer.encoder(src=emb, mask=mask, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, mem_enc, tgt_mask, tgt_padding_mask):
        # embedding + positional encoding
        emb = self.pos_enc(self.tgt_embedding(tgt))

        return self.transformer.decoder(tgt=emb, memory=mem_enc, tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask_src(self, seq):
        seq_len = seq.shape[1]
        seq_mask = torch.zeros((seq_len, seq_len), device=DEVICE).type(torch.bool)
        seq_mask_padding = (seq == src_vocab.pad_id)

        return seq_mask, seq_mask_padding

    def create_mask_tgt(self, seq):
        seq_len = seq.shape[1]
        seq_mask = self.generate_square_subsequent_mask(seq_len)
        seq_mask_padding = (seq == tgt_vocab.pad_id)

        return seq_mask, seq_mask_padding


def greedy_choice_batch(model, src, target_length):
    src = src.to(DEVICE)
    src_mask, src_padding_mask = model.create_mask_src(src)
    src_mask = src_mask.to(DEVICE)

    # save the encoding state for the multiheaded attention
    mem_encode = model.encode(src, src_mask, src_padding_mask).to(DEVICE)

    #init word as start of string (sos token)
    word = torch.ones(src.shape[0], 1).fill_(tgt_vocab.sos_id).type(torch.int).to(DEVICE)

    # padding elem is a 'mask' that is true for all the elements that have achieved the <eos> in the previous iteration
    padding_elem = torch.zeros(src.shape[0], 1).type(torch.bool).to(DEVICE)
    padding_elem = padding_elem.squeeze(1)

    for i in range(target_length - 1):

        #init target masks
        tgt_mask, tgt_padding_mask = model.create_mask_tgt(word)
        tgt_mask = tgt_mask.to(DEVICE)

        # single decoding
        out = model.decode(word, mem_encode, tgt_mask, tgt_padding_mask)

        # linear layer (classifier)
        outlin = model.linear(out[:, -1, :])

        #greedy choice (maximum probability)
        _, next_char = torch.max(outlin, dim=-1)

        # get indices
        index = torch.where(padding_elem == True)[0]

        # second criteria stop if all have achieved <eos>
        if (index.shape[0] >= src.shape[0]):
            break
        else:
            next_char[index] = tgt_vocab.pad_id

            # select the char
            next_char = next_char.to(DEVICE)

            # next_char = next_char.item()
            word = torch.cat([word, next_char.unsqueeze(1)], dim=1)

            # pad all the elements that have achieved the EOS
            padding_elem = padding_elem | (next_char == tgt_vocab.eos_id)

    return word


'''accuracy computation'''
def accuracy(predicted, tgt):
    count = 0
    for elem in range(predicted.shape[0]):
        correct = True
        for i in range(len(predicted[elem, :]) - 1):
            if (predicted[elem, i + 1].item() != tgt[elem, i + 1].item()):
                correct = False

        if (correct):
            count += 1
    accuracy = count / (predicted.shape[0])

    return accuracy

if __name__ == "__main__":
    #HYPERPARAMETERS
    embedding_size = 256
    src_vocab_size = len(src_vocab.id_to_string)
    tgt_vocab_size = len(tgt_vocab.id_to_string)
    nhead = 8
    ffn_hidden_dim = 1024
    n_encoder_layers = 3
    n_dencoder_layers = 2
    check_accuracy = 1000
    accumulating_gradient = 10

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    model = Model(n_encoder_layers, n_dencoder_layers, embedding_size, nhead, src_vocab_size, tgt_vocab_size, ffn_hidden_dim).to(
        DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-9)

    n_epoch = 1
    iteration = 0
    training_loss = []
    validation_loss = []
    iterations = []
    training_accuracy = []
    validation_accuracy = []
    terminate = False
    DESIRED_ACC = 1.0
    while not terminate:
        model.train()
        losses = 0
        batch_iter = 1
        for batch in train_data_loader:
            src = batch[0].to(DEVICE)
            tgt = batch[1].to(DEVICE)

            tgt_input = tgt[:, :-1]
            pred = model(src, tgt_input)

            flat_pred = pred.contiguous().view(-1, pred.shape[-1])

            flat_tgt = tgt[:, 1:].contiguous().view(-1)

            loss = loss_fn(flat_pred, flat_tgt) / accumulating_gradient

            loss.backward()

            losses += loss.item()

            # update gradient
            if (batch_iter % accumulating_gradient == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
                batch_iter = 0

                # check accuracy
            if (iteration % check_accuracy == 0):
                iterations.append(iteration)

                model.eval()
                with torch.no_grad():
                    pred_greedy = greedy_choice_batch(model, src, tgt.shape[1])
                    acc = accuracy(pred_greedy, tgt);
                    training_accuracy.append(acc)
                    training_loss.append(loss.item() * accumulating_gradient)

                print("ITER: ", iteration, " accuracy training = ", acc)
                print("ITER: ", iteration, " loss = ", loss.item(), "\n")

                # evaluation
                with torch.no_grad():
                    acc = 0
                    loss_eval = 0
                    for batch_eval in valid_data_loader:
                        src = batch_eval[0].to(DEVICE)
                        tgt = batch_eval[1].to(DEVICE)

                        tgt_input = tgt[:, :-1]
                        pred = model(src, tgt_input)
                        flat_pred_eval = pred.contiguous().view(-1, pred.shape[-1])
                        flat_tgt_eval = tgt[:, 1:].contiguous().view(-1)
                        loss_eval += loss_fn(flat_pred_eval, flat_tgt_eval).item()
                        pred_greedy = greedy_choice_batch(model, src, tgt.shape[1])
                        acc += accuracy(pred_greedy, tgt);
                    question = []
                    answer = []
                    p = random.randint(0, pred_greedy.shape[0] - 1)
                    for i in range(len(src[p])):
                        question.append(src_vocab.id_to_string[(src[p][i]).item()])

                    print("QUESTION: ", ''.join(question))
                    for i in range(len(pred_greedy[p])):
                        answer.append(tgt_vocab.id_to_string[(pred_greedy[p][i]).item()])
                    print("ANSWER: ", ''.join(answer))
                    validation_accuracy.append(acc / len(valid_data_loader))
                    validation_loss.append(loss_eval / len(valid_data_loader))
                    acc_eval = acc / len(valid_data_loader)
                    if (acc_eval >= DESIRED_ACC):
                        terminate = True
                        break

                print("ITER: ", iteration, " accuracy evaluation = ", acc / len(valid_data_loader), '\n\n')

                model.train()

            batch_iter += 1
            iteration += 1


        print("\n---------END TRAINING ---------\n")

    #Plotting

    plt.figure(1)
    plt.plot(iterations, validation_loss)
    plt.plot(iterations, training_loss, '--')
    plt.legend(["Validation Loss", "Training Loss"])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("loss")
    plt.savefig('loss.png', dpi=300)
    plt.show(block=False)

    plt.figure(2)
    plt.plot(iterations, validation_accuracy)
    plt.plot(iterations, training_accuracy, '--')
    plt.legend(["Validation Accuracy", "Training Accuracy"])
    plt.xlabel("Iterations")
    plt.ylabel("accuracy")
    plt.title("accuracy")
    plt.savefig('accuracy.png', dpi=300)
    plt.show(block=True)
