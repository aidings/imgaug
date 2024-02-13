import numpy as np

def softmax(input, axis=1):
    """softmax

    Args:
        input (np.ndarray): [description]

    Returns:
        [type]: [description]
    """
    input = input - np.max(input, axis=axis, keepdims=True)
    exp = np.exp(input)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def topk(input, k, axis=None, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return val, ind