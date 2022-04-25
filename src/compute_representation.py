import torch
from torch import nn
import cv2
import multiprocessing
import numpy as np
import glob
import os
import warnings
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Torchgan Imports
import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import ParallelTrainer, Trainer

from read_data import *
from betaVAE import *
from gan_utils import *
from fid import *

def get_all_patient_data(path_csv, patch_data_path):
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
    
    # normalize the rna data
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

    all_images = []
    all_rna_data = []
    for _, row in tqdm(train_df.iterrows()):
        WSI = row['wsi_file_name']
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)
        data_path = row['patch_data_path']
        path = os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))
        lmdb_connection = lmdb.open(path,
                                    subdir=False, readonly=True, 
                                    lock=False, readahead=False, meminit=False)
        images = []
        with lmdb_connection.begin(write=False) as lmdb_txn:
            try:
                keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
                idxs = random.sample(list(range(len(keys))), args.sample_size)
                for idx in idxs:
                    try:
                        lmdb_value = lmdb_txn.get(keys[idx])
                        image = decompress_and_deserialize(lmdb_value)
                        images.append(image)
                    except:
                        continue
                
            except Exception as e:
                    print(e)
                    continue
        
        all_images.append(np.asarray(images))
        all_rna_data.append(rna_data)
    
    return all_images, all_rna_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute repersentation of GAN images')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--gan', type=str, default=None,
            help='File with the checkpoint to start with')
    parser.add_argument('--vaegan', type=str, default=None,
            help='File with the second checkpoint to start with')
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument("--multiprocessing",
                      help="Toggle use of multiprocessing for image pre-processing. Defaults to use all cores",
                      default=False,
                      action="store_true")
    parser.add_argument("--sample_size", dest="sample_size",
                      help="Set sample size to use for the computation",
                      type=int)
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                        help='Checkpoint for the vae model')
    parser.add_argument('--path_to_save', type=str, default=None,
                        help='Path to save the models with prefix for the file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(10*'-')
    print('Config for this experiment \n')
    print(config)
    print(10*'-')

    path_csv = config['path_csv']
    patch_data_path = config['patch_data_path']
    img_size = config['img_size']
    max_patch_per_wsi = config['max_patch_per_wsi']
    rna_features = config['rna_features']

    
    betavae = betaVAE(rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=0.0005)
    betavae.load_state_dict(torch.load(args.vae_checkpoint))
    betavae.eval()

    patients_real_images, patients_rna_data = get_all_patient_data(path_csv, patch_data_path)
    
    trainer_gan = load_torchgan_trainer(args.gan)
    trainer_vaegan = load_torchgan_trainer(args.vaegan)
    
    real_activations = []
    gan_activations = []
    vaegan_activations = []

    for real_images, rna_data in tqdm(zip(patients_real_images, patients_rna_data)):
        real_images= preprocess_images(real_images, args.multiprocessing)
        real_act = get_activations(real_images, 1)
        real_activations.append(np.mean(real_act, axis=0))

        fake_images_rna = generate_images(trainer_vaegan, gene_exp=rna_data.unsqueeze(0), sample_size=args.sample_size, betavae=betavae)
        fake_images_rna = preprocess_images(fake_images_rna, args.multiprocessing)
        vaegan_act = get_activations(fake_images_rna, 1)
        vaegan_activations.append(np.mean(vaegan_act, axis=0))

        fake_images = generate_images(trainer_gan,sample_size=args.sample_size)
        fake_images = preprocess_images(fake_images, args.multiprocessing)
        gan_act = get_activations(fake_images, 1)
        gan_activations.append(np.mean(gan_act, axis=0))

    np.save(args.path_to_save+'_real.npy', np.asarray(real_activations))
    np.save(args.path_to_save+'_vaegan.npy', np.asarray(vaegan_activations))
    np.save(args.path_to_save+'_gan.npy', np.asarray(gan_activations))
    
