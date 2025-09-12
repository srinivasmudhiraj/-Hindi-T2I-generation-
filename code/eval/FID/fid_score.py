#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import numpy as np
#from scipy.misc import imread
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as transforms
from eval.FID.inception import InceptionV3
import torch.utils.data
from PIL import Image
from torch.utils import data
import eval.FID.img_data as img_data
import pandas as pd
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
#parser.add_argument('path', type=str, nargs=2,
#                    help=('Path to the generated images or '
#                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--path1', type=str, default=64)
parser.add_argument('--path2', type=str, default=64)

def get_activations(images, model, batch_size=12, dims=2048, cuda=False, verbose=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    #d0 = images.shape[0]

    d0 = images.__len__() * batch_size
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    #for i in range(n_batches):
    for i, batch in enumerate(images):
        #batch = batch[0]
        #if verbose:
            #print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        #import ipdb
        #ipdb.set_trace()
        start = i * batch_size
        end = start + batch_size

        #batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        #batch = Variable(batch, volatile=True)

        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-3):
    """Numpy implementation of the Frechet Distance."""
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Means have different shapes'
    assert sigma1.shape == sigma2.shape, 'Covariance matrices have different shapes'

    # Check for numerical stability of the covariance matrices
    eigenvalues1, _ = np.linalg.eig(sigma1)
    eigenvalues2, _ = np.linalg.eig(sigma2)

    if np.any(eigenvalues1 < 1e-6) or np.any(eigenvalues2 < 1e-6):
        print("Warning: Small eigenvalues detected in covariance matrices.")
    
    # Compute the difference of means
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('Warning: Covariance matrix product is not finite. Adding offset.')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # If there are imaginary components, we discard them
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: Imaginary component {m}. Ignoring it.")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(images, model, batch_size=12,
                                    dims=2048, cuda=False, verbose=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(images, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_FID(paths, writer, epoch, batch_size=12, dims=2048, cuda=True):
    """Compute FID score given two paths."""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    #fid_value = calculate_fid_given_paths(paths, batch_size, cuda, dims)
    #writer.add_scalar('FID', fid_value, epoch)
    #print(f'FID at epoch {epoch}: {fid_value}')
    m1, s1, m2, s2 = calculate_fid_given_paths(paths, batch_size, cuda, dims)

    # Calculate FID score
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"FID value: {fid_value}")
    #save_to_excel(fid_value, epoch)
    return fid_value







def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        print(f.files) 
        m, s = f['mean1'][:], f['covariance1'][:]
        f.close()

    else:
        dataset = img_data.Dataset(path, transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ]))
        print(dataset.__len__())
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=8)
        m, s = calculate_activation_statistics(dataloader, model, batch_size, dims, cuda)
    return m, s
    
    

def calculate_fid_given_paths(paths, batch_size, cuda, dims):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    # Compute statistics for both real and generated images
    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)

    # Return the means and covariances of both datasets
    return m1, s1, m2, s2
##
def save_to_excel(fid_value, epoch, file_path="epoch_scores_fid.xlsx"):
    """
    Save the calculated Inception score to an Excel file across epochs.

    Args:
        mean_score: Calculated mean score.
        std_score: Calculated standard deviation score.
        epoch: The current epoch number.
        file_path: Path to save the results Excel file.
    """
    # Create a DataFrame for every new epoch
    data = {"Epoch": [epoch], "FID_Score": [fid_value]}
    df = pd.DataFrame(data)

    # If file exists, append new data; otherwise, create a new file
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    else:
        df.to_excel(file_path, index=False)

    print(f"Results for epoch {epoch} saved to {file_path}") 

##

if __name__ == '__main__':
    freeze_support()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    paths = ["",""]
    paths[0] = args.path1
    paths[1] = args.path2
    print(paths)
    fid_value = calculate_fid_given_paths(paths, args.batch_size,args.gpu,args.dims)
    print('FID: ', fid_value)
    print('FID: ', fid_value)
