import argparse
import os
import datetime
import json

import torch
import torch.nn as nn
from types_ import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

# Torchgan Imports
import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import ParallelTrainer, Trainer

from wsi_model import *
from read_data import *
from wgan_loss import *
from biggan import BigGanGenerator, BigGanDiscriminator
from sagan import SAGANGenerator, SAGANDiscriminator
from dcgan import DCGANUpGenerator

def custom_collate_fn_wganvae(batch):
    """Remove bad entries from the dataloader

    Args:
        batch (torch.Tensor): batch of tensors from the dataaset

    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x['image'] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader

    Args:
        batch (torch.Tensor): batch of tensors from the dataaset

    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x['image'] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
# https://github.com/torchgan/model-zoo/blob/master/gman/gman.py 
# to update the losses and use the VAE

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='GANs training on histology data')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--checkpoint', type=str, default=None,
            help='File with the checkpoint to start with')
    parser.add_argument('--seed', type=int, default=99,
            help='Seed for random generation')
    parser.add_argument('--image_dir', type=str, default='images',
            help='Image dir to save image')
    parser.add_argument('--model_dir', type=str, default='./model/gan',
            help='Image dir to save model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=None,
            help='Number of epochs to train the model')
    parser.add_argument('--num_patches', type=int,
            help='Number of tiles to use per slide', default=250)
    parser.add_argument('--gan_type', type=str, default='dcgan',
            help='Architecture to use')
    parser.add_argument('--loss_type', type=str, default='wgangp',
            help='Loss type to use')
    args = parser.parse_args()

    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = json.load(f)

    print(10*'-')
    print('Config for this experiment \n')
    print(config)
    print(10*'-')

    if 'flag' in config:
        args.flag = config['flag']
    else:
        args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
    
    if not os.path.exists(args.image_dir):
        os.mkdir(args.image_dir)
    
    path_csv = config['path_csv']
    batch_size = 8
    encoder_checkpoint = config.get('encoder_checkpoint', None)
    patch_data_path = config['patch_data_path']
    save_dir = config['save_dir']
    img_size = config['img_size']
    max_patch_per_wsi = args.num_patches
    quick = False
    bag_size = config.get('bag_size', 40)
    
    print('Number of available GPUs: {}'.format(torch.cuda.device_count()))
    print('Loading dataset...')

    transforms_ = nn.Sequential(
    #transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=(0.5,), std=(0.5,)))

    datasets = []
    for id, (dataset, path) in enumerate(zip(path_csv, patch_data_path)):
        print(dataset)
        data_paths_train = []
        data_paths_val = []
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

    if args.loss_type == 'wganvae':
        print(f"Using {args.loss_type}")
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

    if args.loss_type == 'wganvae':
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                num_workers=4, pin_memory=True, collate_fn=custom_collate_fn_wganvae)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # training
    print('Finished loading dataset and creating dataloader')

    print('Initializing models')
    
    if args.gan_type == 'dcgan':
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
    elif args.gan_type == 'condgan':
        generator = ConditionalGANGenerator
        discriminator = ConditionalGANDiscriminator
        arguments_generator = {
                "encoding_dims": 2048,
                "out_channels": 3,
                "step_channels": 32,
                "out_size": 256,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.Tanh(),
            }
        arguments_discriminator = {
                "in_size": 256,
                "in_channels": 3,
                "step_channels": 32,
                "nonlinearity": nn.LeakyReLU(0.2),
                "last_nonlinearity": nn.LeakyReLU(0.2),
            }
    elif args.gan_type == 'biggan':
        generator = BigGanGenerator
        discriminator = BigGanDiscriminator
        arguments_generator = {
            "encoding_dims": 2048,
            "out_channels": 3,
            "step_channels": 32,
            "out_size": 256,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh(),
            "G_ch":64,
            "dim_z":2048,
            "resolution":256,
            "n_classes": 2,
        }
        arguments_discriminator = {
            "resolution":256,
            "n_classes": 1,
             "in_size": 256,
            "in_channels": 3,
            "step_channels": 32,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2),
        }
    elif args.gan_type == 'sagan':
        generator = SAGANGenerator
        discriminator = SAGANDiscriminator
        arguments_generator = {
                "encoding_dims": 2048,
                "step_channels": 32
            }
        arguments_discriminator = {
                "step_channels": 32,
            }
    else:
        raise "Model proposed not implemented"
    
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

    if args.gan_type == 'condgan':
        gan_network['generator']['args']['num_classes'] = len(path_csv)
        gan_network['discriminator']['args']['num_classes'] = len(path_csv)

    if args.loss_type == 'minimax':
        losses = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
    elif args.loss_type == 'wgan':
        losses = [
            WassersteinGeneratorLoss(),
            WassersteinDiscriminatorLoss(clip=(-0.01, 0.01)),
            WassersteinGradientPenalty(),
        ]
    elif args.loss_type == 'wganvae':
        losses = [
            WassersteinGeneratorLossVAE(checkpoint='checkpoints/betavae_training_tissues/model_dict_best.pt', rna_features=19198),
            WassersteinDiscriminatorLossVAE(checkpoint='checkpoints/betavae_training_tissues/model_dict_best.pt', rna_features=19198),
            WassersteinGradientPenaltyVAE(checkpoint='checkpoints/betavae_training_tissues/model_dict_best.pt', rna_features=19198),
        ]
    elif args.loss_type == 'lsgan':
        losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]

    else:
        assert f"Loss type {args.loss_type} not implemented. \
                Choose between minimax, wgangp, lsgan or wganvae."

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Use deterministic cudnn algorithms
        torch.backends.cudnn.deterministic = True
        epochs = args.num_epochs
    else:
        device = torch.device("cpu")
        epochs = 5

    print("Device: {}".format(device))
    print("Epochs: {}".format(epochs))

    trainer = Trainer(
    gan_network, losses, checkpoints=args.model_dir,
    sample_size=64, epochs=epochs, devices=[0],
    recon=args.image_dir
    )

    """
    Snippet for generating data:
        generator = getattr(trainer, "generator")
        test_noise = getattr(trainer, "generator").sampler(trainer.sample_size, trainer.device)
        images = generator(test_noise[0])
    """
    
    if args.checkpoint is not None:
        trainer.load_model(load_path=args.checkpoint)
    
    trainer(train_dataloader)
