from matplotlib.animation import ImageMagickBase
import torch
from torch import nn
from torchvision.models import inception_v3
import cv2
import multiprocessing
import numpy as np
import glob
import os
from scipy import linalg
import warnings
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from tqdm import tqdm

# Torchgan Imports
import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import ParallelTrainer, Trainer

from read_data import *
from betaVAE import *
from gan_utils import *

"""
Implementation of the FID metric from: https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/c4bef90e502e7e1aec2a1a8f45b259630e093f8b/fid.py#L162
with my own modificatiopns
"""

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def get_activations(images, batch_size, device='cuda:0'):
    """
    Calculates activations for last pool layer for all iamges
    --
        Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch size: batch size used for inception network
    --
    Returns: np array shape: (N, 2048), dtype: np.float32
    """
    assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                              ", but got {}".format(images.shape)

    num_images = images.shape[0]
    inception_network = PartialInceptionNetwork()
    inception_network = inception_network.to(device)
    inception_network.eval()
    n_batches = int(np.ceil(num_images  / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = ims.to(device)
        activations = inception_network(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
        inception_activations[start_idx:end_idx, :] = activations
    return inception_activations



def calculate_activation_statistics(images, batch_size):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer 
                of the inception model.
    """
    act = get_activations(images, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def preprocess_image(im):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
    Return:
        im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
    """
    
    assert im.shape[2] == 3
    assert len(im.shape) == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    im = cv2.resize(im, (299, 299))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im)
    
    assert im.max() <= 1.0
    assert im.min() >= 0.0
    assert im.dtype == torch.float32
    assert im.shape == (3, 299, 299)

    return im


def preprocess_images(images, use_multiprocessing):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
    """
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for im in images:
                job = pool.apply_async(preprocess_image, (im,))
                jobs.append(job)
            final_images = torch.zeros(images.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                im = job.get()
                final_images[idx] = im#job.get()
    else:
        final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float32
    return final_images


def calculate_fid(images1, images2, use_multiprocessing, batch_size):
    """ Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    images1 = preprocess_images(images1, use_multiprocessing)
    images2 = preprocess_images(images2, use_multiprocessing)
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute FID')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--config2', type=str, help='JSON config file',
                         default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
            help='File with the checkpoint to start with')
    parser.add_argument('--checkpoint2', type=str, default=None,
            help='File with the second checkpoint to start with')
    parser.add_argument('--patient1', type=str, default=None,
            help='Patient id to select images')
    parser.add_argument('--patient2', type=str, default=None,
            help='Patient id to select images')
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument("--multiprocessing",
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    parser.add_argument("--sample_size", dest="sample_size",
                      help="Set sample size to use for the computation",
                      type=int)
    parser.add_argument("--vae",
                      help="If the data wants to be generated conditioned on the gene expression",
                      default=False,
                      action="store_true")
    parser.add_argument("--real",
                      help="If the real data wants to be used",
                      default=False,
                      action="store_true")
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                        help='Checkpoint for the vae model')
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    print('Values:')
    print(f'Checkpoint 1 {args.checkpoint}')
    print(f'Checkpoint 2 {args.checkpoint2}')
    print(f'VAE {args.vae}')
    print(f'Patient 1 {args.patient1}')
    print(f'Patient 2 {args.patient2}')
    print('\n')

    path_csv = config['path_csv']
    patch_data_path = config['patch_data_path']
    img_size = config['img_size']
    max_patch_per_wsi = config['max_patch_per_wsi']
    rna_features = config['rna_features']

    if args.vae:
        betavae = betaVAE(rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=0.0005)
        betavae.load_state_dict(torch.load(args.vae_checkpoint))
        betavae.eval()
    
    if args.patient1 != None:
        real_images1, rna_data1 = load_images_from_patient(path_csv, patch_data_path, img_size, max_patch_per_wsi, 
                                                            args.patient1, batch_size=args.sample_size, vae=args.vae)
    if args.patient2 != None:
        if args.config2:
            with open(args.config2) as f:
                config = json.load(f)
            path_csv = config['path_csv']
            patch_data_path = config['patch_data_path']

        real_images2, rna_data2 = load_images_from_patient(path_csv, patch_data_path, img_size, max_patch_per_wsi, 
                                                            args.patient2, batch_size=args.sample_size, vae=args.vae)
    else:
        real_images, _ = load_images(path_csv, patch_data_path, img_size, max_patch_per_wsi, batch_size=args.sample_size)
    
    trainer = load_torchgan_trainer(args.checkpoint)
    if args.checkpoint2:
        trainer2 = load_torchgan_trainer(args.checkpoint2)
    
    fid_values = []

    iterations = 5
    for i in range(iterations):
        if args.vae:
            fake_images = generate_images(trainer, gene_exp=rna_data1, sample_size=args.sample_size, betavae=betavae)
        else:
            fake_images = generate_images(trainer, sample_size=args.sample_size)
        
        if args.checkpoint2:
            fake_images2 = generate_images(trainer2, sample_size=args.sample_size)
            fid_value = calculate_fid(fake_images, fake_images2, args.multiprocessing, 2)
        else:
            if args.patient1 != None and args.patient2 != None and args.real:
                fid_value = calculate_fid(real_images1, real_images2, args.multiprocessing, 2)
            else:
                fid_value = calculate_fid(real_images, fake_images, args.multiprocessing, 2)
        fid_values.append(fid_value)

    print(f"FID values {fid_values}\n")
    print(f"The FID value is {np.mean(fid_values)}+-{np.std(fid_values)}")
