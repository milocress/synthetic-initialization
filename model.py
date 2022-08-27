from typing import Dict, List, Union
import torch
import numpy as np
from string import ascii_lowercase
import torch.nn.functional as F

default_tokens = {c: x+1 for x, c in enumerate(ascii_lowercase)}
default_tokens['_'] = 0 # padding character


def tokenize(input_string: Union[str, List[str]], token_mapping: Dict[str, int] = None):
    if token_mapping is None:
        token_mapping = default_tokens
    if isinstance(input_string, list):
        return torch.tensor([[token_mapping[c] for c in reversed(s)] for s in input_string]), len(token_mapping)

    return torch.tensor([token_mapping[c] for c in reversed(input_string)]), len(token_mapping)


def decode(tokens, skip_special_tokens=True):
    default_decoding = {x: c for x, c in enumerate(ascii_lowercase)}
    default_decoding[len(default_decoding)] = '' if skip_special_tokens else '_' 
    return ''.join([default_decoding[token.item()] for token in reversed(list(tokens))])

def grab_tokens(logits):
    F.softmax(logits.transpose(1, 2), dim=2)

def make_weights_set(alphabet_size: int, sequence_length: int):
    """Generates weights for set"""
    embedding_length = alphabet_size + sequence_length + 2

    helper_tensor = torch.ones((alphabet_size, alphabet_size)) * -1.0
    helper_tensor.fill_diagonal_(1)
    flag_tensor = torch.zeros((alphabet_size, 2))
    position_tensor = torch.zeros((alphabet_size, sequence_length))
    helper_tensor = torch.cat((flag_tensor, helper_tensor, position_tensor), dim=1)

    helper_tensor_2 = torch.ones((sequence_length, sequence_length)) * -2
    helper_tensor_2.fill_diagonal_(1.0)
    helper_tensor_2.triu_(0)
    zero_tensors = torch.zeros((sequence_length, alphabet_size + 2))
    helper_tensor_2 = torch.cat((zero_tensors, helper_tensor_2), dim=1)

    weight_tensor = torch.cat((helper_tensor, helper_tensor_2), dim=0)
    weight_tensor = torch.cat((torch.eye(embedding_length), weight_tensor))
    rand_tensor = torch.randn((2 * embedding_length + 2, embedding_length))
    weight_tensor = torch.cat((weight_tensor, rand_tensor), dim=0)

    weight_tensor_2 = torch.zeros((embedding_length, embedding_length * 4))
    weight_tensor_2.fill_diagonal_(1)
    flag_neuron_1 = torch.zeros((1, embedding_length * 4))
    flag_neuron_2 = torch.zeros((1, embedding_length * 4))

    flag_neuron_1[0, embedding_length : alphabet_size + embedding_length] = 1
    flag_neuron_2[0, alphabet_size + embedding_length : alphabet_size + sequence_length + embedding_length] = 1

    weight_tensor_2[0] = flag_neuron_1
    weight_tensor_2[1] = flag_neuron_2

    return weight_tensor, weight_tensor_2

def make_weights_id(embedding_size: int):
    in_dim, hidden_dim = embedding_size, embedding_size * 4
    weight_tensor_1 = torch.randn((hidden_dim - in_dim, in_dim))
    weight_tensor_1 = torch.cat((torch.eye(in_dim), weight_tensor_1), dim=0)
    weight_tensor_2 = torch.zeros((in_dim, hidden_dim))
    weight_tensor_2.fill_diagonal_(1)
    return weight_tensor_1, weight_tensor_2

class BasicSelfAttention(torch.nn.Module):
    def __init__(self, softmax_enabled: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax_enabled = softmax_enabled

    def forward(self, X):
        raw_weights = torch.bmm(X, X.transpose(1, 2))
        weights = F.softmax(raw_weights, dim=2) if self.softmax_enabled else raw_weights
        y = torch.bmm(weights, X).float()
        return y

class SingleHeadAttention(torch.nn.Module):
    def __init__(self, dim: int, softmax_enabled: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax_enabled = softmax_enabled
        self.dim = dim

        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)

    def set_weights(self, task: str):
        with torch.no_grad():
            self.k.weight.data = torch.eye(self.dim).float()
            self.q.weight.data = torch.eye(self.dim).float()
            self.v.weight.data = torch.eye(self.dim).float()

    def forward(self, X):
        k = self.k(X)
        q = self.q(X)
        v = self.v(X)

        dot = torch.bmm(q, k.transpose(1, 2))

        y = torch.bmm(dot, v)
        return y


class FeedForwardRelu(torch.nn.Module):
    def __init__(self, alphabet_size: int, sequence_length: int):
        super().__init__()

        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length

        embedding_length = alphabet_size + sequence_length + 2
        self.embedding_length = embedding_length
        in_dim, hidden_dim = embedding_length, embedding_length*4
        out_dim = in_dim

        self.layer0 = torch.nn.Linear(in_dim, hidden_dim, bias=False)
        self.relu   = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(hidden_dim, out_dim, bias=False)


    def forward(self, X):
        return self.layer1(self.relu(self.layer0(X)))

    def set_weights(self, task: str):
        with torch.no_grad():
            if task == 'set':
                weight_tensor_0, weight_tensor_1 = make_weights_set(self.alphabet_size, self.sequence_length)
                self.layer0.weight.data = weight_tensor_0
                self.layer1.weight.data = weight_tensor_1
            elif task == 'id':
                weight_tensor_0, weight_tensor_1 = make_weights_id(self.embedding_length)
                self.layer0.weight.data = weight_tensor_0
                self.layer1.weight.data = weight_tensor_1
            else:
                raise ValueError(f"unknown task {task}")


class ExtendedEmbeddings(torch.nn.Module):
    def __init__(self, alphabet_size: int, sequence_length: int):
        super().__init__()

        embedding_length = alphabet_size + sequence_length + 2
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.token_embedding = torch.nn.Embedding(alphabet_size, embedding_dim=embedding_length)
        self.position_embedding = torch.nn.Embedding(sequence_length, embedding_dim=embedding_length)

    def forward(self, X):
        batch_size = X.shape[0]
        P = torch.arange(0, self.sequence_length, 1).repeat(batch_size, 1)
        return self.token_embedding(X) + self.position_embedding(P)

    def set_weights(self, task: str):
        with torch.no_grad():
            flag_columns = torch.zeros((self.alphabet_size, 2))
            eye = torch.eye(self.alphabet_size)
            position_columns = torch.zeros((self.alphabet_size, self.sequence_length))
            self.token_embedding.weight.data = torch.cat((
                flag_columns, 
                eye,
                position_columns
                ), dim=1)
            self.token_embedding.weight.data[:,0] = 0
            self.token_embedding.weight.data[:,1] = 0
            self.token_embedding.weight.data[0,0] = 2
            self.token_embedding.weight.data[0,1] = -2
            flag_columns = torch.zeros((self.sequence_length, 2))
            eye = torch.eye(self.sequence_length)
            token_columns = torch.zeros((self.sequence_length, self.alphabet_size))
            self.position_embedding.weight.data = torch.cat((
                flag_columns, 
                token_columns,
                eye,
                ), dim=1)


class BasicTransformerBlock(torch.nn.Module):
    def __init__(self, alphabet_size: int, sequence_length: int) -> None:
        super().__init__()
        
        embedding_length = alphabet_size + sequence_length + 2
        self.em = ExtendedEmbeddings(alphabet_size, sequence_length)
        self.sa = SingleHeadAttention(embedding_length)
        self.ff = FeedForwardRelu(alphabet_size, sequence_length)

    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        X = self.em(input_ids)
        X = self.sa(X)
        X = self.ff(X)
        X = X.transpose(1,2)
        W = self.em.token_embedding.weight.unsqueeze(0).expand(batch_size, *self.em.token_embedding.weight.shape)
        prob_logits = torch.bmm(W, X)
        return prob_logits

    def set_weights(self, task: str):
        self.em.set_weights(task)
        self.sa.set_weights(task)
        self.ff.set_weights(task)

# TODO:
# - find short seq-seq finetuning and run training
# - prosper
