import torch
import cv2
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from read_data import *
from betaVAE import *
from gan_utils import *

import matplotlib.pyplot as plt

def savegrid(ims, rows=None, cols=None, fill=True, showax=False, figname='test.png'):
    if rows is None != cols is None:
        raise ValueError("Set either both rows and cols or neither.")

    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                 axes_pad=0.0,  # pad between axes in inch.
                 )

    for ax, im in zip(grid, ims):
    # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_axis_off()
    '''
    if rows is None:
        rows = len(ims)
        cols = 1

    gridspec_kw = {'wspace': 0, 'hspace': 0} if fill else {}
    fig,axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)

    if fill:
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax,im in zip(axarr.ravel(), ims):
        ax.imshow(im)
        if not showax:
            ax.set_axis_off()
    '''
    #kwargs = {'pad_inches': .01} if fill else {}
    plt.tight_layout()
    fig.savefig(figname, dpi=300)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate new images based on given gene expression')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--checkpoint', type=str, default=None,
            help='File with the checkpoint to start with')
    parser.add_argument('--checkpoint2', type=str, default=None,
            help='File with the checkpoint to start with')
    parser.add_argument('--patient1', type=str, default=None,
            help='Patient id to select images')
    parser.add_argument("--sample_size", dest="sample_size",
                      help="Set sample size to use for the computation",
                      type=int)
    parser.add_argument("--rna_features", dest="rna_features",
                      help="number of genes to use",
                      default=19198,
                      type=int)
    parser.add_argument("--rna_file",
                      help="RNA file containing the gene expression",
                      default=None,
                      type=str)
    parser.add_argument('--vae_checkpoint', type=str, default=None,
                        help='Checkpoint for the vae model')
    parser.add_argument("--save_path",
                      help="Filename to save the generated images",
                      default=None,
                      type=str)
    parser.add_argument('--random_patient', action='store_true', default=False)
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

    betavae = betaVAE(args.rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=0.0005)
    betavae.load_state_dict(torch.load(args.vae_checkpoint))
    betavae.eval()

    trainer = load_torchgan_trainer(args.checkpoint)

    if args.random_patient:
        rna_file = pd.read_csv(args.rna_file)
        rna_data = rna_file.sample(1).values
        rna_data = torch.from_numpy(rna_data).to(torch.float32)
        fake_images = generate_images(trainer, gene_exp=rna_data, sample_size=args.sample_size, betavae=betavae)
        savegrid(fake_images, rows=8, cols=8, figname=args.save_path)
    else:
        real_images1, rna_data1 = load_images_from_patient(path_csv, patch_data_path, img_size, max_patch_per_wsi, 
                                                             args.patient1, batch_size=args.sample_size, vae=args.vae)

        trainer = load_torchgan_trainer(args.checkpoint)
        trainer2 = load_torchgan_trainer(args.checkpoint2)

        fake_images = generate_images(trainer, gene_exp=rna_data1, sample_size=args.sample_size, betavae=betavae)
        fake_images2 = generate_images(trainer2, sample_size=args.sample_size)
        i=0
        for img1, img2, img3 in zip(real_images1, fake_images, fake_images2):
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 *= 255
            img2 = img2.astype(np.uint8)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            img3 *= 255
            img3 = img3.astype(np.uint8)
            img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/real_brain_{}.png'.format(args.save_dir,i), img1)
            cv2.imwrite('{}/vaegan_brain_{}.png'.format(args.save_dir,i), img2)
            cv2.imwrite('{}/gan_brain_{}.png'.format(args.save_dir,i), img3)
            i += 1

