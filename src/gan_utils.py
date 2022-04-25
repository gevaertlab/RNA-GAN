import torch
from torch import nn
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

# Torchgan Imports
import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import Trainer

from read_data import *
from betaVAE import *

def collate_fn(batch):
    """Remove bad entries from the dataloader

    Args:
        batch (torch.Tensor): batch of tensors from the dataaset

    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
    
def decompress_and_deserialize(lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except:
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def load_images(path_csv, patch_data_path, img_size, max_patch_per_wsi,
                batch_size=64, quick=False, vae=False):
    """ Loads all .png or .jpg images from a given path
    Warnings: Expects all images to be of same dtype and shape.
    Args:
        path: relative path to directory
    Returns:
        final_images: np.array of image dtype and shape.
    """
    transforms_ = nn.Sequential(
    #transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float))
    #transforms.Normalize(mean=(0.5,), std=(0.5,)))

    datasets = []
    for id, (dataset, path) in enumerate(zip(path_csv, patch_data_path)):
        print(dataset)
        data_paths_train = []
        df = pd.read_csv(dataset)
        data_paths_train = [path] * df.shape[0]
        df['patch_data_path'] = data_paths_train
        label = [id] * df.shape[0]
        df['labels'] = label

        datasets.append(df)
        
    if(len(datasets) >=2):
        train_df = pd.concat([datasets[0], datasets[1]])
        for i in range(2, len(datasets)):
            train_df = pd.concat([train_df, datasets[i]])
    else:
        train_df = datasets[0]
    
    if vae:
        def _get_log(x):
                # trick to take into account zeros
                x = np.log(x.replace(0, np.nan))
                return x.replace(np.nan, 0)
        
        # get list of columns to scale
        rna_columns = [x for x in train_df.columns if 'rna_' in x]
        non_rna_columns = [x for x in train_df.columns if 'rna_' not in x]
        # log transform
        train_df[rna_columns] = train_df[rna_columns].apply(_get_log)
        
        train_df = train_df[rna_columns+non_rna_columns]
        
        rna_values = train_df[rna_columns].values

        scaler = StandardScaler()
        rna_values = scaler.fit_transform(rna_values)

        train_df[rna_columns] = rna_values

        train_dataset = PatchRNADataset(patch_data_path, train_df, img_size,
                            max_patches_total=max_patch_per_wsi,
                            transforms=transforms_,
                            quick=quick)
    else:
        train_dataset = PatchDataset(patch_data_path, train_df, img_size,
                            max_patches_total=max_patch_per_wsi,
                            transforms=transforms_,
                            quick=quick)

    train_dataloader = DataLoader(train_dataset,  collate_fn=collate_fn, batch_size=batch_size, 
                num_workers=10, pin_memory=True)

    batch = next(iter(train_dataloader))
    real_images = torch.permute(batch[0], (0, 2, 3, 1)).numpy()
    rna_data = None
    if vae:
        rna_data = batch[1]
    return real_images, rna_data

def load_images_from_patient(path_csv, patch_data_path, img_size, max_patch_per_wsi,
                patient_path, batch_size=64, quick=False, vae=True):
    """ Loads all .png or .jpg images from a given path
    Warnings: Expects all images to be of same dtype and shape.
    Args:
        path: relative path to directory
    Returns:
        final_images: np.array of image dtype and shape.
    """
    transforms_ = nn.Sequential(
    #transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float))
    #transforms.Normalize(mean=(0.5,), std=(0.5,)))

    datasets = []
    for id, (dataset, path) in enumerate(zip(path_csv, patch_data_path)):
        print(dataset)
        data_paths_train = []
        df = pd.read_csv(dataset)
        data_paths_train = [path] * df.shape[0]
        df['patch_data_path'] = data_paths_train
        label = [id] * df.shape[0]
        df['labels'] = label

        datasets.append(df)
        
    if(len(datasets) >=2):
        train_df = pd.concat([datasets[0], datasets[1]])
        for i in range(2, len(datasets)):
            train_df = pd.concat([train_df, datasets[i]])
    else:
        train_df = datasets[0]
    
    
    def _get_log(x):
        # trick to take into account zeros
        x = np.log(x.replace(0, np.nan))
        return x.replace(np.nan, 0)
    
    # get list of columns to scale
    rna_columns = [x for x in train_df.columns if 'rna_' in x]
    non_rna_columns = [x for x in train_df.columns if 'rna_' not in x]
    # log transform
    train_df[rna_columns] = train_df[rna_columns].apply(_get_log)
    
    train_df = train_df[rna_columns+non_rna_columns]
    
    rna_values = train_df[rna_columns].values

    scaler = StandardScaler()
    rna_values = scaler.fit_transform(rna_values)

    train_df[rna_columns] = rna_values

    row = train_df.loc[train_df['wsi_file_name'] == patient_path]
    WSI = row['wsi_file_name'].values[0]
    rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
    rna_data = torch.tensor(rna_data, dtype=torch.float32)
    data_path = row['patch_data_path'].values[0]
    path = os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))
    lmdb_connection = lmdb.open(path,
                                subdir=False, readonly=True, 
                                lock=False, readahead=False, meminit=False)
    images = []
    with lmdb_connection.begin(write=False) as lmdb_txn:
        keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
        idxs = random.sample(list(range(len(keys))), batch_size)
        for idx in idxs:
            try:
                lmdb_value = lmdb_txn.get(keys[idx])
                image = decompress_and_deserialize(lmdb_value)
                images.append(image)
            except:
                pass
    
    return np.asarray(images), rna_data

def generate_images(trainer, gene_exp=None, sample_size=64, betavae=None):
    """Generate images using pre-trained gan

    Args:
        trainer (torchgan.Trainer): Trainer from the torchgan framework.
        gene_exp (torch.Tensor): Tensor representing the gene expression of the patient
        sample_size (int): Number of samples to generate
    Returns:
        Tensor: Images generated by the GAN
    """
    generator = getattr(trainer, "generator")
    generator = generator.to(trainer.device)
    if gene_exp != None:
        betavae = betavae.to(trainer.device)
        noise = torch.FloatTensor(sample_size, generator.encoding_dims).uniform_(-0.3, 0.3)
        noise = noise.to(trainer.device)
        z, _, _ = betavae.encode(gene_exp.to(trainer.device))
        z = z.detach().to(trainer.device)
        noise = noise + z
        noise = (noise - torch.mean(noise, dim=0)) / torch.std(noise, dim=0)
        new_noise = torch.split(noise,10)
        images = []
        for i in range(len(new_noise)):
            im = generator(new_noise[i])
            images.append(im.detach().cpu().numpy())
        images = np.concatenate(images, axis=0)
        images = torch.from_numpy(images)
        images = images.view((-1, 3, 256, 256))
    else:
        test_noise = getattr(trainer, "generator").sampler(sample_size, trainer.device)
        new_noise = torch.split(test_noise[0],10)
        images = []
        for i in range(len(new_noise)):
            im = generator(new_noise[i])
            images.append(im.detach().cpu().numpy())
        images = np.concatenate([images], axis=0)
        images = torch.from_numpy(images)
        images = images.view((-1, 3, 256, 256))

    mean = torch.tensor([0.5,0.5,0.5])
    std = torch.tensor([0.5,0.5,0.5])

    unnormalize_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    images = unnormalize_trans(images)
    images = torch.permute(images, (0, 2, 3, 1))
    images = images.detach().cpu().numpy()

    return images

def load_torchgan_trainer(checkpoint):
    """Load torchgan trained for the generation

    Args:
        checkpoint (str): File with the trainer parameters.

    Returns:
        [torcgan.Trainer]: Trained loadded from the checkpoint
    """
    generator = DCGANGenerator
    discriminator = DCGANDiscriminator
    arguments_generator = {
                "encoding_dims": 2048,
                "out_channels": 3,
                "step_channels": 64,
                "out_size": 256,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            }
    arguments_discriminator = {
            "in_size": 256,
            "in_channels": 3,
            "step_channels": 64,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2),
        }
    
    gan_network = {
        "generator": {
            "name": generator,
            "args": arguments_generator,
            "optimizer": {"name": Adam, "args": {"lr": 0.0001, "betas": (0.5, 0.999)}},
        },
        "discriminator": {
            "name": discriminator,
            "args": arguments_discriminator,
            "optimizer": {"name": Adam, "args": {"lr": 0.0004, "betas": (0.5, 0.999)}},
        },
    }

    losses = [
            WassersteinGeneratorLoss(),
            WassersteinDiscriminatorLoss(),
            WassersteinGradientPenalty(),
        ]
    
    trainer = Trainer(
    gan_network, losses, checkpoints="test",
    sample_size=64, epochs=1, devices=[0,1],
    recon="images"
    )
    trainer.load_model(load_path=checkpoint)

    return trainer