import numpy as np
import random

# import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from quantization import cosine_quantization, cosine_dequantization
from quantization import linear_quantization, linear_dequantization 

import sys

class VirtualWorker():
    def __init__(self, wid):
        self.wid = wid
        #self.dset = None 
        self.loader = None
        self.opt = None

    def set_loader(self, loader):
        self.loader = loader

    def set_opt(self, opt):
        self.opt = opt


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_model(src, dst):
    for s, d in zip( src.parameters() , dst.parameters() ):
        d.data = s.data.detach().clone()

def update_model(model, buffer, args):
    q_option = args.quantize_option
    n_bits = args.quantize_bits
    sparse = args.sparse

    for k, p in model.named_parameters(): 
        weight = 600 * len(buffer['gradient_data'])
        grad_out = 0
        n_nan = 0
        
        #print(k)
        for i in range(len(buffer['gradient_data'])):
            # TODO: check whether the name of the current grad exists in the buffer
            if not k in buffer['gradient_data'][i].keys():
                break
            data = buffer['gradient_data'][i][k]

            if q_option == 'none':
                data = data
            elif q_option == 'cosine':
                data = cosine_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k])
            elif q_option == 'linear':
                data = linear_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k], buffer['gradient_rec3'][i][k], args.quantize_hadamard)
            else:
                print("Unexpected quantization method:", sys.exc_info()[0])
                raise RuntimeError from OSError

            if sparse == 1:
                grad_out += - data * 600 / weight
            elif sparse > 0 and sparse < 1:
                mask = (torch.rand(data.size()) < sparse).type(data.type()).cuda()
                grad_out += - data * mask  / sparse * 600 / weight
            else:
                print("Unexpected sparsification ratio:", sys.exc_info()[0])
                raise RuntimeError from OSError

        if args.n_update_client > n_nan:
            p.data.add_( grad_out.cuda() )
'''

def dec_grad_update_model(model, buffer, args, prefix=None):
    q_option = args.quantize_option
    n_bits = args.quantize_bits
    sparse = args.sparse
    for k, p in model.named_parameters(): 
        weight = 600 * len(buffer['gradient_data'])
        grad_out = 0
        n_nan = 0
        
        if prefix is not None:
            k = prefix + '.' + k
        #print(k)
        for i in range(len(buffer['gradient_data'])):
            # TODO: check whether the name of the current grad exists in the buffer
            if not k in buffer['gradient_data'][i].keys():
                break
            data = buffer['gradient_data'][i][k]

            if q_option == 'none':
                data = data
            elif q_option == 'cosine':
                data = cosine_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k])
            elif q_option == 'linear':
                data = linear_dequantization(data, n_bits, buffer['gradient_rec1'][i][k], buffer['gradient_rec2'][i][k], buffer['gradient_rec3'][i][k], args.quantize_hadamard)
            else:
                print("Unexpected quantization method:", sys.exc_info()[0])
                raise RuntimeError from OSError

            if sparse == 1:
                grad_out += - data * 600 / weight
            elif sparse > 0 and sparse < 1:
                mask = (torch.rand(data.size()) < sparse).type(data.type()).cuda()
                grad_out += - data * mask  / sparse * 600 / weight
            else:
                print("Unexpected sparsification ratio:", sys.exc_info()[0])
                raise RuntimeError from OSError

        if args.n_update_client > n_nan:
            p.data.add_( grad_out )

def update_model(model, buffer, args):
    dec_grad_update_model(model.cur_model, buffer, args, prefix='cur_model')
    dec_grad_update_model(model.head, buffer, args, prefix='head')
'''
def compute_client_gradients(model, model_new, buffer, args):
    q_option = args.quantize_option
    q_clip = args.quantize_clip
    n_bits = args.quantize_bits

    gradient_data = {}
    gradient_rec1 = {}
    gradient_rec2 = {}
    gradient_rec3 = {}

    for m1, m2 in zip( model.named_parameters() , model_new.named_parameters() ):
        assert m1[0] == m2[0]
        assert m1[1].shape == m2[1].shape
        #print(m1[0], m2[0])
        tmp = m1[1] - m2[1]
        # cast the gradients from gpus to cpus
        tmp = tmp
        
        if q_option == 'none':
            gradient_data[m1[0]] = tmp
        elif q_option == 'cosine':
            quantized_grad, norm_grad, bound_grad = cosine_quantization(tmp, n_bits, q_clip)
            gradient_data[m1[0]] = quantized_grad
            gradient_rec1[m1[0]] = norm_grad
            gradient_rec2[m1[0]] = bound_grad
        elif q_option == 'linear':
            quantized_grad, min_grad, max_grad, diag_grad = linear_quantization(tmp, n_bits, args.quantize_unbiased, args.quantize_hadamard)
            gradient_data[m1[0]] = quantized_grad
            gradient_rec1[m1[0]] = min_grad
            gradient_rec2[m1[0]] = max_grad
            gradient_rec3[m1[0]] = diag_grad
        else:
            print("Unexpected quantization method:", sys.exc_info()[0])
            raise RuntimeError from OSError
    
    buffer['gradient_data'].append(gradient_data)
    buffer['gradient_rec1'].append(gradient_rec1)
    buffer['gradient_rec2'].append(gradient_rec2)
    buffer['gradient_rec3'].append(gradient_rec3)

def noniid(dataset, num_users, shard_per_user, rand_set_all=[]):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = len(np.unique(dataset.targets))
    shard_per_class = int(shard_per_user * num_users / num_classes)
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        assert(len(x)) <= shard_per_user
        test.append(value)
    test = np.concatenate(test)
    assert(len(test) == len(dataset))
    assert(len(set(list(test))) == len(dataset))

    return dict_users, rand_set_all
