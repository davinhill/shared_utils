import os
import sys
import pickle
from tqdm import tqdm

import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader

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

def exAUC(torch_samples, attributions, ref_samples, model, pct_exclude = np.linspace(0,1,21).tolist()):
    '''
    '''
    # get labels, i.e. model prediction with all features
    dataloader = create_dataloader(data = torch_samples, labels = None)
    labels = batch_predictions(model, dataloader)

    # flatten samples and attributions
    orig_shape = torch_samples.shape
    torch_samples = torch.flatten(torch_samples, 1).cpu()
    ref_samples = torch.flatten(ref_samples, 1).cpu()
    if type(attributions == np.ndarray):
        attributions = np.abs(attributions)
        attributions = attributions.reshape(orig_shape[0],-1)
    else:
        attributions = torch.abs(attributions)
        attributions = torch.flatten(attributions, 1)

    #iteratively mask features according to attribution ranking
    output_list = []
    for j, k in tqdm(enumerate(pct_exclude), total = (len(pct_exclude)+1)):

        num_feats = attributions.shape[1]
        n_exclude = min(int(num_feats * k), num_feats) # number of features to mask
        torch_samples_tmp = torch_samples.clone().cpu()
        for i, (torch_sample, attribution) in enumerate(zip(torch_samples, attributions)):
            attribution = np.abs(attribution[:num_feats])
            ranking = np.argsort(attribution)
            
            if n_exclude == 0:
                continue # original sample only
            elif n_exclude == num_feats:
                torch_samples_tmp[i,:] = ref_samples[i,:]
            else:
                torch_samples_tmp[i, ranking[-n_exclude:]] = ref_samples[i, ranking[-n_exclude:]]

        dataloader = create_dataloader(torch_samples_tmp.reshape(orig_shape), labels)
        output_list.append(calc_test_accy(model, dataloader))
            
    output_list = np.array(output_list)
    auc = sklearn.metrics.auc(x = pct_exclude, y = output_list)
    return auc, output_list



def incAUC(torch_samples, attributions, ref_samples, model, pct_include = np.linspace(0,1,21).tolist()):
    '''
    '''
    # get labels, i.e. model prediction with all features
    dataloader = create_dataloader(data = torch_samples, labels = None)
    labels = batch_predictions(model, dataloader)

    # flatten samples and attributions
    orig_shape = torch_samples.shape
    torch_samples = torch.flatten(torch_samples, 1).cpu()
    ref_samples = torch.flatten(ref_samples, 1).cpu()
    if type(attributions == np.ndarray):
        attributions = np.abs(attributions)
        attributions = attributions.reshape(orig_shape[0],-1)
    else:
        attributions = torch.abs(attributions)
        attributions = torch.flatten(attributions, 1)

    #iteratively mask features according to attribution ranking
    output_list = []
    for j, k in tqdm(enumerate(pct_include), total = (len(pct_include)+1)):

        num_feats = attributions.shape[1]
        n_include = min(int(num_feats * k), num_feats) # number of features to mask
        ref_sample_tmp = ref_samples.clone().cpu()
        for i, (torch_sample, attribution) in enumerate(zip(torch_samples, attributions)):
            ranking = np.argsort(np.abs(attribution))
            
            if n_include == 0:
                continue # reference sample only
            elif n_include == num_feats:
                ref_sample_tmp[i,:] = torch_sample
            else:
                ref_sample_tmp[i, ranking[-n_include:]] = torch_sample[ranking[-n_include:]]

        dataloader = create_dataloader(ref_sample_tmp.reshape(orig_shape), labels)
        output_list.append(calc_test_accy(model, dataloader))
            
    output_list = np.array(output_list)
    auc = sklearn.metrics.auc(x = pct_include, y = output_list)
    return auc, output_list

def create_dataloader(data, labels, batch_size = 64):
    dataset = plain_dataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

class plain_dataset(Dataset):
    def __init__(self, data, labels):
        """
        args:
            samples: torch tensor of samples
        """
        self.data = data
        if labels is None:
            self.labels = torch.zeros(self.data.shape[0])
        else:
            self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx,...], self.labels[idx]

def calc_test_accy(model, test_loader):
    model.eval()   # Set model into evaluation mode
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = tensor2cuda(data), tensor2cuda(target)
            output = model(data)   # Calculate Output
            try:
                dim_output = output.shape[1]
            except IndexError:
                dim_output = 1
            if dim_output == 1:
                pred = (output>=0)*1
            else:
                pred = output.max(1, keepdim=True)[1]  # Calculate Predictions

            correct += pred.eq(target.view_as(pred)).sum().item()
        return (100.*correct/len(test_loader.dataset))

def batch_predictions(model, test_loader):
    model.eval()   # Set model into evaluation mode
    pred_list = []
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = tensor2cuda(data), tensor2cuda(target)
            output = model(data)   # Calculate Output
            try:
                dim_output = output.shape[1]
            except IndexError:
                dim_output = 1
            if dim_output == 1:
                pred = (output>=0)*1
            else:
                pred = output.max(1, keepdim=True)[1]  # Calculate Predictions

            pred_list.append(pred.detach().cpu())
    
    return torch.cat(pred_list).reshape(-1)

