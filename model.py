import torch
import numpy as np
from numpy.typing import NDArray
from string import ascii_lowercase

default_tokens = {c: x for x, c in enumerate(ascii_lowercase)}


def tokenize(input_string: str, token_mapping: str = None):
    if token_mapping is None:
        token_mapping = default_tokens
    return np.array([token_mapping[c] for c in input_string]), len(token_mapping)


def batch_tokens(tokens: NDArray):
    """batch tokens if not already batched"""
    if len(tokens.shape) == 1:
        return np.array([tokens])
    elif len(tokens.shape) == 2:
        return tokens

    raise ValueError("too many dimensions")


def generate_position_embeddings(batch_size, sequence_length: int):
    """ "one-hot position embeddings"""
    M = np.eye(sequence_length)
    return torch.tensor(M).repeat(batch_size, 1, 1)


def embed(tokens, num_values=None):
    """ "Generates embeddings for a batch of tokens with an alphabet size of ``num_values``.
    Embeddings are one-hot on the alphabet and one-hot on positions"""
    batch_size, sequence_length = tokens.shape
    num_values = num_values + 2 if num_values is not None else np.max(tokens) + 1 + 2

    one_hot = torch.tensor(np.eye(num_values)[tokens + 2])
    positional = generate_position_embeddings(batch_size, sequence_length)

    return torch.cat((one_hot, positional), dim=2)


def make_weights(alphabet_size: int, sequence_length: int):
    """Generates weights"""
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

class BasicSelfAttention(torch.nn.Module):
    def __init__(self, softmax_enabled: bool = False):
        super.__init__()
        self.softmax_enabled = softmax_enabled

    def forward(self, X):
        raw_weights = torch.bmm(X, X.transpose(1, 2))
        weights = F.softmax(raw_weights, dim=2) if self.softmax_enabled else raw_weights
        y = torch.bmm(weights, X).float()
        return y


class FeedForwardRelu(torch.nn.Module):
    def __init__(self, alphabet_size: int, sequence_length: int):
        super.__init__()

        embedding_length = alphabet_size + sequence_length + 2
        in_dim, hidden_dim = embedding_length, embedding_length*4
        out_dim = in_dim

        self.layer0 = torch.nn.Linear(in_dim, hidden_dim)
        self.relu   = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(hidden_dim, out_dim)

        weight_tensor_0, weight_tensor_1 = make_weights(alphabet_size, sequence_length)
        self.layer0.weight = weight_tensor_0
        self.layer1.weight = weight_tensor_1

    def forward(self, X):
        return self.layer1(self.relu(self.layer0(X)))


class BasicTransformerBlock(torch.nn.Module):
    def __init__(self, alphabet_size: int, sequence_length: int) -> None:
        super().__init__()

        embedding_length = alphabet_size + sequence_length + 2

        self.sa = BasicSelfAttention()
        self.ln = torch.nn.LayerNorm(embedding_length)
        self.ff = FeedForwardRelu(alphabet_size, sequence_length)

    def forward(self, X):
        X = self.sa(X)
        X = self.ff(X)
        X = self.ln(X)

