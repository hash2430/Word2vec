from abc import ABC
from typing import List, Dict, Tuple, Set
import random

### You may import any Python standard library here.
import torch
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
### END YOUR LIBRARIE

import torch
from dataset import SkipgramDataset

def naive_softmax_loss(
    center_vectors: torch.Tensor, outside_vectors: torch.Tensor,
    center_word_index: torch.Tensor, outside_word_indices: torch.Tensor
) -> torch.Tensor:
    """ Naive softmax loss function for word2vec models

    Implement the naive softmax losses between a center word's embedding and an outside word's embedding.
    When using GPU, it is efficient to perform a large calculation at once, so batching is used generally.
    In addition, using a large batch size reduces the variance of samples in SGD, making training process more effective and accurate.
    To practice this, let's calculate batch-sized losses of skipgram at once.
    <PAD> tokens are appended for batching if the number of outside words is less than 2 * window_size.
    However, these arbitrarily inserted <PAD> tokens have no meaning so should NOT be included in the loss calculation.

    !!!IMPORTANT: Do NOT forget eliminating the effect of <PAD> tokens!!!

    Note: Try not to use 'for' iteration as you can. It may degrade your performance score. You can complete this file without any for iteration.
    Use built-in functions in pyTorch library. They must be faster than your hard-coded script. You can use any funtion in pyTorch library.

    Hint: torch.index_select function would be helpful

    Arguments:
    center_vectors -- center vectors is
                        in shape (num words in vocab, word vector length)
                        for all words in vocab (V in the pdf handout)
    outside_vectors -- outside vector is
                        in shape (num words in vocab, word vector length)
                        for all words in vocab (U in the pdf handout)
    center_word_index -- the index of the center word
                        in shape (batch size,)
                        (c of v_c in the pdf handout)
    outside_word_indices -- the indices of the outside words
                        in shape (batch size, window size * 2)
                        (all o of u_o in the pdf handout.
                        <PAD> tokens are inserted for padding if the number of outside words is less than window size * 2)

    Return:
    losses -- naive softmax loss for each (center_word_index, outsied_word_indices) pair in a batch
                        in shape (batch size,)
    """
    assert center_word_index.shape[0] == outside_word_indices.shape[0]

    n_tokens, word_dim = center_vectors.shape
    batch_size, outside_word_size = outside_word_indices.shape
    PAD = SkipgramDataset.PAD_TOKEN_IDX

    ### YOUR CODE HERE (~4 lines)
    # W: outside vector matrix for denominator
    W = torch.index_select(outside_vectors, 0, torch.tensor(range(1,n_tokens)).to(device))

    # Exclude <PAD> token
    mask_2D = ~outside_word_indices.ne(0)
    non_pad_cnt = mask_2D.sum(1)

    # Z: center word vector matrices
    Z = center_vectors.index_select(0, center_word_index).unsqueeze(2)

    # X: outside vector matrix for numerator
    outside_word_indices_flat = outside_word_indices.view(batch_size*outside_word_size)
    X = outside_vectors.index_select(0, outside_word_indices_flat).view(batch_size, outside_word_size, word_dim)

    # Calculate numerator
    numerator = X.bmm(Z).exp().squeeze()

    # Calculate denominator
    denominator = Z.squeeze().matmul(W.transpose(1,0)).exp().sum(1).unsqueeze(1).expand(-1, outside_word_size)

    # Calculate loss
    divided = numerator/denominator
    divided[mask_2D] = 1
    losses = (-1)*torch.log(divided).view(batch_size, outside_word_size).sum(1)
    # losses = losses/non_pad_cnt.to(torch.float32)
    ### END YOUR CODE
    assert losses.shape == torch.Size([batch_size])
    return losses


def neg_sampling_loss_5min(
    center_vectors: torch.Tensor, outside_vectors: torch.Tensor,
    center_word_index: torch.Tensor, outside_word_indices: torch.Tensor,
    negative_sampler, K: int=10
) -> torch.Tensor:
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss for each pair of (center_word_index, outside_word_indices) in a batch.
    As same with naive_softmax_loss, all inputs are batched with batch_size.

    !!!IMPORTANT: Do NOT forget eliminating the effect of <PAD> tokens!!!

    Note: Implementing negative sampler is a quite tricky job so we pre-implemented this part. See below comments to check how to use it.
    If you want to know how the sampler works, check SkipgramDataset.negative_sampler code in dataset.py file

    Hint: torch.gather function would be helpful

    Arguments/Return Specifications: same as naiveSoftmaxLoss

    Additional arguments:
    negative_sampler -- the negative sampler
    K -- the number of negative samples to take
    """
    assert center_word_index.shape[0] == outside_word_indices.shape[0]

    n_tokens, word_dim = center_vectors.shape
    batch_size, outside_word_size = outside_word_indices.shape
    PAD = SkipgramDataset.PAD_TOKEN_IDX

    ##### Sampling negtive indices #####
    # Because each outside word needs K negatives samples,
    # negative_sampler takes a tensor in shape [batch_size, outside_word_size] and gives a tensor in shape [batch_size, outside_word_size, K]
    # where values in last dimension are the indices of sampled negatives for each outside_word.
    negative_samples: torch.Tensor = negative_sampler(outside_word_indices, K)
    assert negative_samples.shape == torch.Size([batch_size, outside_word_size, K])

    ###  YOUR CODE HERE (~5 lines)
    losses: torch.Tensor = None
    mask_X = ~outside_word_indices.ne(0)
    negative_samples[outside_word_indices.unsqueeze(-1).expand(-1, -1, K) == 0] = 0
    mask_W = ~negative_samples.ne(0)
    # center_word_index_flat = center_word_index.repeat(outside_word_size).view(outside_word_size, batch_size).transpose(1,0).flatten()
    outside_word_indices_flat = outside_word_indices.view(batch_size*outside_word_size)
    negative_word_indices_flat = negative_samples.flatten()

    Z = center_vectors.index_select(0, center_word_index).unsqueeze(2)
    X = outside_vectors.index_select(0, outside_word_indices_flat).view(batch_size, outside_word_size, word_dim)
    W = outside_vectors.index_select(0, negative_word_indices_flat).view(batch_size, outside_word_size * K, word_dim)

    left_term = X.bmm(Z).squeeze()
    left_term[mask_X] = float('inf')
    left_term = torch.sigmoid(left_term)
    left_term = torch.log(left_term)

    right_term = (-1) * W.bmm(Z).view(batch_size, outside_word_size, K)
    right_term[mask_W] = float('inf')
    right_term = torch.sigmoid(right_term)
    right_term = torch.log(right_term)
    right_term = right_term.sum(dim=2)

    losses = (-1) * (left_term + right_term).sum(1)


    ### END YOUR CODE
    assert losses.shape == torch.Size([batch_size])
    return losses

def neg_sampling_loss(
    center_vectors: torch.Tensor, outside_vectors: torch.Tensor,
    center_word_index: torch.Tensor, outside_word_indices: torch.Tensor,
    negative_sampler, K: int=10
) -> torch.Tensor:
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss for each pair of (center_word_index, outside_word_indices) in a batch.
    As same with naive_softmax_loss, all inputs are batched with batch_size.

    !!!IMPORTANT: Do NOT forget eliminating the effect of <PAD> tokens!!!

    Note: Implementing negative sampler is a quite tricky job so we pre-implemented this part. See below comments to check how to use it.
    If you want to know how the sampler works, check SkipgramDataset.negative_sampler code in dataset.py file

    Hint: torch.gather function would be helpful

    Arguments/Return Specifications: same as naiveSoftmaxLoss

    Additional arguments:
    negative_sampler -- the negative sampler
    K -- the number of negative samples to take
    """
    assert center_word_index.shape[0] == outside_word_indices.shape[0]

    n_tokens, word_dim = center_vectors.shape
    batch_size, outside_word_size = outside_word_indices.shape
    PAD = SkipgramDataset.PAD_TOKEN_IDX

    ##### Sampling negtive indices #####
    # Because each outside word needs K negatives samples,
    # negative_sampler takes a tensor in shape [batch_size, outside_word_size] and gives a tensor in shape [batch_size, outside_word_size, K]
    # where values in last dimension are the indices of sampled negatives for each outside_word.
    negative_samples: torch.Tensor = negative_sampler(outside_word_indices, K)
    assert negative_samples.shape == torch.Size([batch_size, outside_word_size, K])

    ###  YOUR CODE HERE (~5 lines)
    mask_X = ~outside_word_indices.ne(0)
    negative_samples[mask_X.unsqueeze(-1).expand(-1, -1, K)] = 0
    mask_W = ~negative_samples.ne(0)
    outside_word_indices_flat = outside_word_indices.view(batch_size*outside_word_size)
    negative_word_indices_flat = negative_samples.flatten()

    Z = center_vectors.index_select(0, center_word_index).unsqueeze(2)
    X = outside_vectors.index_select(0, outside_word_indices_flat).view(batch_size, outside_word_size, word_dim)
    W = outside_vectors.index_select(0, negative_word_indices_flat).view(batch_size, outside_word_size * K, word_dim)

    XW = torch.cat((X, W),dim=1)
    mult = XW.bmm(Z)
    left_term = mult[:,:outside_word_size,:].squeeze()
    left_term[mask_X] = float('inf')
    left_term = torch.sigmoid(left_term).log()

    right_term = (-1) * mult[:,outside_word_size:,:].view(batch_size, outside_word_size, K)
    right_term[mask_W] = float('inf')
    right_term = torch.sigmoid(right_term).log().sum(dim=2)

    losses = (-1) * (left_term + right_term).sum(1)
    ### END YOUR CODE
    assert losses.shape == torch.Size([batch_size])
    return losses

#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class Word2Vec(torch.nn.Module, ABC):
    """
    A helper class that wraps your word2vec losses.
    """
    def __init__(self, n_tokens: int, word_dimension: int):
        super().__init__()

        self.center_vectors = torch.nn.Parameter(torch.empty([n_tokens, word_dimension])) # V
        self.outside_vectors = torch.nn.Parameter(torch.empty([n_tokens, word_dimension])) # U

        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.center_vectors.data)
        torch.nn.init.normal_(self.outside_vectors.data)

class NaiveWord2Vec(Word2Vec):
    def forward(self, center_word_index: torch.Tensor, outside_word_indices: torch.Tensor):
        return naive_softmax_loss(self.center_vectors, self.outside_vectors, center_word_index, outside_word_indices)

class NegSamplingWord2Vec(Word2Vec):
    def __init__(self, n_tokens: int, word_dimension: int, negative_sampler, K: int=10):
        super().__init__(n_tokens, word_dimension)

        self._negative_sampler = negative_sampler
        self._K = K

    def forward(self, center_word_index: torch.Tensor, outside_word_indices: torch.Tensor):
        return neg_sampling_loss(self.center_vectors, self.outside_vectors, center_word_index, outside_word_indices, self._negative_sampler, self._K)

#############################################
# Testing functions below.                  #
#############################################

def naive_softmax_loss_test():
    print ("======Naive Softmax Loss Test Case======")
    center_word_index = torch.randint(1, 100, [10]).to(device)
    outside_word_indices = []
    for _ in range(10):
        random_window_size = random.randint(3, 6)
        outside_word_indices.append([random.randint(1, 99) for _ in range(random_window_size)] + [0] * (6 - random_window_size))
    outside_word_indices = torch.Tensor(outside_word_indices).to(torch.long).to(device)

    model = NaiveWord2Vec(n_tokens=100, word_dimension=3).to(device)

    loss = model(center_word_index, outside_word_indices).mean()
    loss.backward()

    # first test
    assert (model.center_vectors.grad[0, :] == 0).all() and (model.outside_vectors.grad[0, :] == 0).all(), \
        "<PAD> token should not affect the result."
    print("The first test passed! Howerver, this test doesn't guarantee you that <PAD> tokens really don't affects result.")

    # Second test
    temp = model.center_vectors.grad.clone().detach()
    temp[center_word_index] = 0.
    assert (temp == 0.).all() and (model.center_vectors.grad[center_word_index] != 0.).all(), \
        "Only batched center words can affect the center_word embedding."
    print("The second test passed!")

    # third test
    assert loss.detach().allclose(torch.tensor(26.86926651).to(device)), \
        "Loss of naive softmax do not match expected result."
    print("The third test passed!")

    # forth test
    expected_grad = torch.Tensor([[-0.07390384, -0.14989397,  0.03736909],
                                  [-0.00191219,  0.00386495, -0.00311787],
                                  [-0.00470913,  0.00072215,  0.00303244]]).to(device)
    assert model.outside_vectors.grad[1:4, :].allclose(expected_grad), \
        "Gradients of naive softmax do not match expected result."
    print("The forth test passed!")

    print("All 4 tests passed!")

def neg_sampling_loss_test():
    print ("======Negative Sampling Loss Test Case======")
    center_word_index = torch.randint(1, 100, [5])
    outside_word_indices = []
    for _ in range(5):
        random_window_size = random.randint(3, 6)
        outside_word_indices.append([random.randint(1, 99) for _ in range(random_window_size)] + [0] * (6 - random_window_size))
    outside_word_indices = torch.Tensor(outside_word_indices).to(torch.long)

    neg_sampling_prob = torch.ones([100])
    neg_sampling_prob[0] = 0.

    dummy_database = type('dummy', (), {'_neg_sample_prob': neg_sampling_prob})

    sampled_negatives = list()
    def negative_sampler_wrapper(outside_word_indices, K):
        result = SkipgramDataset.negative_sampler(dummy_database, outside_word_indices, K)
        sampled_negatives.clear()
        sampled_negatives.append(result)
        return result

    model = NegSamplingWord2Vec(n_tokens=100, word_dimension=3, negative_sampler=negative_sampler_wrapper, K=5)

    loss = model(center_word_index, outside_word_indices).mean()
    loss.backward()

    # first test
    assert (model.center_vectors.grad[0, :] == 0).all() and (model.outside_vectors.grad[0, :] == 0).all(), \
        "<PAD> token should not affect the result."
    print("The first test passed! Howerver, this test dosen't guarantee you that <PAD> tokens really don't affects result.")    

    # Second test
    temp = model.center_vectors.grad.clone().detach()
    temp[center_word_index] = 0.
    assert (temp == 0.).all() and (model.center_vectors.grad[center_word_index] != 0.).all(), \
        "Only batched center words can affect the centerword embedding."
    print("The second test passed!")

    # Third test
    sampled_negatives = sampled_negatives[0]
    sampled_negatives[outside_word_indices.unsqueeze(-1).expand(-1, -1, 5) == 0] = 0
    affected_indices = list((set(sampled_negatives.flatten().tolist()) | set(outside_word_indices.flatten().tolist())) - {0})
    temp = model.outside_vectors.grad.clone().detach()
    temp[affected_indices] = 0.
    assert (temp == 0.).all() and (model.outside_vectors.grad[affected_indices] != 0.).all(), \
        "Only batched outside words and sampled negatives can affect the outside word embedding."
    print("The third test passed!")

    # forth test
    # assert loss.detach().allclose(torch.tensor(35.82903290)), \
    #     "Loss of negative sampling do not match expected result."
    # print("The forth test passed!")

    # fifth test
    expected_grad = torch.Tensor([[ 0.08583137, -0.40312022, -0.05952500],
                                  [ 0.14896543, -0.53478962, -0.18037169],
                                  [ 0.03650964,  0.24137473, -0.21831468]])
    assert model.outside_vectors.grad[affected_indices[:3], :].allclose(expected_grad), \
        "Gradient of negative sampling do not match expected result."
    print("The fifth test passed!")

    print("All 5 tests passed!")


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    torch.manual_seed(4321)
    random.seed(4321)

    naive_softmax_loss_test()
    neg_sampling_loss_test()
