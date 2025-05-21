import os
import sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from itertools import permutations, chain, combinations
import math

#######################################################################
#  **   *******  
# /**  **/////** 
# /** **     //**
# /**/**      /**
# /**/**      /**
# /**//**     ** 
# /** //*******  
# //   ///////   
#######################################################################

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_args(args):
    for x, y in vars(args).items():
        print('{:<16} : {}'.format(x, y))

def merge_dicts(dictionary_list):
    """
    merge a list of of dictionaries into a single dictionary
    """
    output = {}
    for dictionary in dictionary_list:
        output.update(dictionary)
    return output

def chdir_script(file):
    '''
    Changes current directory to that of the current python script

    args:
        file: "__file__"
    '''
    abspath = os.path.abspath(file)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def get_filedir(file):
    '''
    returns directory path of current file

    args:
        file: "__file__"
    '''
    abspath = os.path.abspath(file)
    dname = os.path.dirname(abspath)
    return dname

def make_dir_fromfile(filepath):
    dirpath = os.path.dirname(filepath)
    make_dir(dirpath)

def list_fonts():
    '''
    print fonts available in system
    '''
    import matplotlib.font_manager
    fpaths = matplotlib.font_manager.findSystemFonts()

    for i in fpaths:
        f = matplotlib.font_manager.get_font(i)
        print(f.family_name)

def dict_to_argparse(dictionary):
    '''
    converts a dictionary of variables and values to argparse format

    input:
        dictionary
    return:
        argparse object
    '''
    import argparse
    parser = argparse.ArgumentParser()
    for k, v in dictionary.items():
        parser.add_argument('--' + k, default = v)

    args, unknown = parser.parse_known_args()
    return args




#######################################################################
#  ****     ** **     ** ****     **** *******  **    **
# /**/**   /**/**    /**/**/**   **/**/**////**//**  ** 
# /**//**  /**/**    /**/**//** ** /**/**   /** //****  
# /** //** /**/**    /**/** //***  /**/*******   //**   
# /**  //**/**/**    /**/**  //*   /**/**////     /**   
# /**   //****/**    /**/**   /    /**/**         /**   
# /**    //***//******* /**        /**/**         /**   
# //      ///  ///////  //         // //          //    
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x * (x>0)

def softplus(x):
    return np.log(1+np.exp(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def exp_kernel_func(mat, lam=0.5, q=2):
    '''
    elementwise exp(-lam * mat^q)

    input:
        mat: matrix of distances
        lam: lambda
        q: q

    '''
    return np.exp(-lam * (mat ** q))

def flatten(array, dim = 1):
    '''
    equivalent to torch.flatten for numpy 

    args:
        array (np or tensor)
        start_dim: function will flatten all dimensions after start_dim
    '''

    input_shape = array.shape
    return array.reshape(list(input_shape[:dim]) + [-1])


def np_insert(matrix, vector, index):
    '''
    insert vector into matrix (as column) at index
    '''
    matA = matrix[:, :index]
    matB = matrix[:, index:]
    return np.concatenate((matA, vector, matB), axis = 1)

def np_collapse(matrix, index):
    '''
    remove column from matrix
    '''
    matA = matrix[:, :index]
    matB = matrix[:, index+1:]
    return np.concatenate((matA, matB), axis = 1)

def subsample_rows(matrix1, max_rows, matrix2 = None, seed = 0):
    '''
    randomly samples rows of a matrix if the number of rows is greater than max_rows

    args:
        matrix: matrix to be sampled.
        max_rows: maximum number of rows that the matrix should contain
    
    return:
        subsampled matrix
    '''
    n_rows = matrix1.shape[0]
    np.random.seed(seed)
    if n_rows > max_rows:
        idx = np.random.choice(n_rows, size = max_rows, replace = False)
        matrix1 = matrix1[idx,...]
        if matrix2 is not None:
            matrix2 = matrix2[idx,...]
            return (matrix1, matrix2)
        else:
            return matrix1
    else:
        if matrix2 is not None:
            return (matrix1, matrix2)
        else:
            return matrix1

def binary_mc2sc_modelwrapper(model):
    '''
    wrapper to convert the 2d output (n x 2) of binary classifiers to 1d (n) by selecting dimension 1 of the output.
    '''
    def wrapper(*args, **kwargs):
        output = model(*args, **kwargs)
        return output[:,1]
    return wrapper

def superpixel_transform_matrix(n_features, superpixel_size):
    '''
    converts a n_features x n_features matrix to superpixel x superpixel matrix by summing adjacent pixels.

    return:
        binary matrix of superpixel x n_features

    '''
    n_superpixels = np.ceil(n_features / superpixel_size).astype('int')
    t_matrix = np.zeros((n_features, n_superpixels)) # matrix to transform feature attributions
    for i in range(n_superpixels):
        grouped_features = np.arange(i * superpixel_size, min(i * superpixel_size + superpixel_size, n_features))
        t_matrix[grouped_features,i] = 1
    return t_matrix

def invert_permutation(p):
    '''
    Given a permutation p of a vector [1,...len(p)], return the inverse permutation s such that p[s] = np.arange(len(p))
    '''
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def invert_permutation_matrix(p):
    '''
    Given a permutation p of a vector [1,...len(p)], return the inverse permutation s such that p[s] = np.arange(len(p))

    Input is a matrix of permutations. Output is a matrix of inverse permutations.
    '''
    if type(p) == np.ndarray:
        output = np.empty_like(p)
        mapping = np.arange(p.shape[1]).reshape(1,-1).repeat(p.shape[0], axis = 0)
        for i in range(output.shape[0]):
            output[i,p[i,:]] = mapping[i,:]
        return output
    elif type(p) == torch.Tensor:
        output = torch.zeros_like(p)
        return output.scatter_(1, p, torch.arange(p.size(1)).expand(p.size(0),-1))

def sample_permutation_3d(n_samples, n_features, n_permutations, reduce_perm_samples = False):
    '''
    sample n_samples permutations for vector {1,...,n_features}
    
    args:
        n_samples: (int) number of samples
        n_features: (int) number of features
        n_permutations: (int) number of permutations to sample
        reduce_perm_samples: (bool) if True, check if total number of permutations is less than n_permutations. If so, reduce n_permutations to the total number of permutations.

    returns: 
        samples x features x n_permutations numpy matrix
    '''

    if reduce_perm_samples and math.factorial(n_features) <= n_permutations: # if number of permutations is less than n_perm_samples, calculate all permutations
        perm_list = list(permutations(range(n_features)))
        n_permutations = len(perm_list)

    samples = np.zeros((n_samples,n_features, n_permutations)).astype('int')
    for i in range(n_samples):
        if reduce_perm_samples and math.factorial(n_features) <= n_permutations:
            samples[i,...] = np.array(perm_list).transpose()
        else:
            samples[i,...] = sample_permutation(n_permutations, n_features).transpose()
    return samples

def sample_permutation(n, d, reduce_perm_samples = False):
    '''
    sample n permutations for vector {1,...,d}
    
    args:
        n: (int) number of permutations
        d: (int) dimension
        reduce_perm_samples: (bool) if True, check if total number of permutations is less than n_permutations. If so, reduce n_permutations to the total number of permutations.

    returns: 
        n x d numpy matrix
    '''

    if reduce_perm_samples and math.factorial(d) <= n: # if number of permutations is less than n_perm_samples, calculate all permutations and reduce the number of returned samples
        perm_list = list(permutations(range(d)))
        n = len(perm_list)

    samples = np.zeros((n,d)).astype('int')

    if math.factorial(d) == n: # if number of permutation samples == total number of permutations, return all permutations
        perm_list = list(permutations(range(d)))
        samples = np.array(perm_list)
    else:
        for i in range(n):
            samples[i,:] = np.random.permutation(d)
    return samples


def list_permutations(vector):
    '''
    list all permutations of a numpy vector
    '''
    output = list(permutations(vector))
    output = [np.array(x) for x in output]
    return output

def list_power_set(vector):
    '''
    Given a numpy vector, list all possible subsets of the vector
    '''

    subsets = list(chain.from_iterable(combinations(vector, r) for r in range(len(vector) + 1)))
    subsets = [np.array(subset) for subset in subsets]
    return subsets

def invert_permutation_subset(vector):
    '''
    Given a permutation p of a vector of positive integers, return the inverse permutation s. Does not assume that the vector contains all integers from 0 to n-1.
    '''
    # tmp_vector = np.arange(vector.max()+1)
    tmp_vector = np.sort(vector)
    output = np.zeros_like(np.arange(vector.max()+1))
    output[vector] = tmp_vector
    return output[tmp_vector]


#######################################################################
#  **********   *******   *******     ******  **      **
# /////**///   **/////** /**////**   **////**/**     /**
#     /**     **     //**/**   /**  **    // /**     /**
#     /**    /**      /**/*******  /**       /**********
#     /**    /**      /**/**///**  /**       /**//////**
#     /**    //**     ** /**  //** //**    **/**     /**
#     /**     //*******  /**   //** //****** /**     /**
#     //       ///////   //     //   //////  //      // 
######################################################################

def tensor2numpy(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    return x

def list2cuda(list, cuda = True):
    array = np.array(list)
    return numpy2cuda(array, cuda = cuda)

def numpy2cuda(array, cuda = True):
    tensor = torch.from_numpy(array)
    return tensor2cuda(tensor, cuda = cuda)

def tensor2cuda(tensor, cuda = True):
    if torch.cuda.is_available() and cuda:
        tensor = tensor.cuda()
    return tensor

def auto2cuda(obj):
    # Checks object type, then calls corresponding function
    if type(obj) == list:
        return list2cuda(obj)
    elif type(obj) == np.ndarray:
        return numpy2cuda(obj)
    elif type(obj) == torch.Tensor:
        return tensor2cuda(obj)
    else:
        raise ValueError('input must be list, np array, or pytorch tensor')

def auto2numpy(obj):
    # Checks object type, then calls corresponding function
    if type(obj) == list:
        return np.array(obj)
    elif type(obj) == np.ndarray:
        return obj
    elif type(obj) == torch.Tensor:
        return tensor2numpy(obj)
    else:
        raise ValueError('np array, or pytorch tensor')

def batch2cuda(list):
    # Input: list of objects to convert. Iterates auto2cuda over list
    output_list = []
    for obj in list:
        output_list.append(auto2cuda(obj))
    return output_list


def idx_to_binary(index_tensor, n_cols = None):
    '''
    Converts vector of indices, where each element of the vector corresponds to a column index in each row of a matrix, into a binary matrix.

    args:
        index_tensor: d-dimensional vector
    returns:
        d x d binary matrix
    '''

    # save input type
    source_type = type(index_tensor)
    if source_type == torch.Tensor:
        device = index_tensor.device
    else:
        device = None

    index_tensor = auto2cuda(index_tensor)

    if n_cols is None:
        n_cols = index_tensor.max().item()+1

    '''
    index_tensor = index_tensor.reshape(-1,1)
    output = tensor2cuda(torch.zeros((index_tensor.shape[0], n_cols), dtype = torch.int32))
    source = tensor2cuda(torch.ones_like(index_tensor, dtype = output.dtype))

    output = output.scatter(dim = 1, index = index_tensor, src = source)
    '''

    output = F.one_hot(index_tensor, num_classes = n_cols)

    # revert type to match input
    if device is not None:
        output = output.to(device)
    if source_type == np.ndarray:
        output = tensor2numpy(output)


    return output


def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return min(lr)

def relu2softplus(model, softplus_beta = 1):
    '''
    Given a Pytorch model, convert all ReLU functions to Softplus
    '''
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta = softplus_beta, threshold = 20))
        else:
            relu2softplus(child)

def freeze_layers(model):
    # freeze all weights in pytorch model
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_layers(model):
    # unfreeze all weights in pytorch model
    for parameter in model.parameters():
        parameter.requires_grad = True

def convert_dataparallel(state_dict):
    '''
    when wrapping a model with nn.DataParallel, the layers are prepended with 'module.'.
    This function 1) checks if layers are prepended with 'module.', then 2) removes this prepended layer name.
    '''
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        prepend = k[:7]
        if prepend == 'module.':
            name = k[7:] # remove 'module.'
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def exp_kernel_func(mat, lam=0.5, q=2, scaling = 0):
    '''
    elementwise exp(-lam * mat^q)

    input:
        mat: matrix of distances
        lam: lambda
        q: q

    '''
    return torch.exp(-lam * (mat ** q) + scaling)

def normalize_l2(mat, dim=1):
    '''
    given a matrix m, normalize all elements such that the L2 norm along dim equals 1

    input:
        mat: matrix
        dim: dimension to normalize

    '''
    norm = torch.norm(mat, dim = dim)
    return torch.div(mat, norm.unsqueeze(dim).expand(mat.shape))

def normalize_01(mat, dim=1):
    '''
    given a matrix m, normalize all elements such that the L2 norm along dim to be within [0,1]

    input:
        mat: matrix
        dim: dimension to normalize

    '''
    min_value = torch.min(mat, dim = dim)[0].unsqueeze(dim).expand(mat.shape)
    max_value = torch.max(mat, dim = dim)[0].unsqueeze(dim).expand(mat.shape)
    norm = max_value - min_value

    return (mat - min_value) / norm


def downsample_images(images, downsample_factor):
    '''
    n x c x h x w matrix. assumes h == w.
    '''

    images = numpy2cuda(images)
    images = images.sum(dim = 1) # sum over channels
    m = torch.nn.AvgPool2d(downsample_factor, stride=downsample_factor, divisor_override=1)
    return tensor2numpy(m(images))