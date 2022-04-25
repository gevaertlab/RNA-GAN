import os
import json
import argparse
import datetime
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from model import *
from betaVAE import *
from read_data import *
from utils import *

parser = argparse.ArgumentParser(description='betaVAE disantangle generation of RNA-Seq data')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--checkpoint', type=str, default=None,
        help='File with the checkpoint to start with')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--parallel', type=int, default=None,
        help='If data parallel wants to be used for training')
parser.add_argument('--type', type=str, default='tissue',
        help='Type of interpolation to do: tissue or sex')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

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

if not os.path.exists(config['save_dir']):
    os.mkdir(config['save_dir'])

path_csv = config['path_csv']
rna_features = config['rna_features']
batch_size = config.get('batch_size', 64)
encoder_checkpoint = config.get('encoder_checkpoint', None)
beta = config.get('beta', 2)
quick = config.get('quick', 0)
opt = config.get('optimizer', 'Adam')

print('Loading dataset...')

datasets = {
    'train': [],
    'test': [],
    'val': []
}
for dataset in path_csv:
    print(dataset)
    df = pd.read_csv(dataset)
    train_df, test_df = train_test_split(df, test_size=0.2)

    train_df, val_df = train_test_split(train_df, test_size=0.2)

    #train_df, val_df, test_df, scaler = normalize_dfs(train_df, val_df, test_df, norm_type='minmax')

    datasets['train'].append(train_df)
    datasets['test'].append(test_df)
    datasets['val'].append(val_df)

if(len(datasets['train']) >=2):
    train_df = pd.concat([datasets['train'][0], datasets['train'][1]])
    val_df = pd.concat([datasets['val'][0], datasets['val'][1]])
    test_df = pd.concat([datasets['test'][0], datasets['test'][1]])
    for i in range(2, len(datasets['train'])):
        print(i)
        train_df = pd.concat([train_df, datasets['train'][i]])
        val_df = pd.concat([val_df, datasets['val'][i]])
        test_df = pd.concat([test_df, datasets['test'][i]])
else:
    train_df = datasets['train'][0]
    val_df = datasets['val'][0]
    test_df = datasets['test'][0]


print('Train shape {}'.format(train_df.shape))
print('Val shape {}'.format(val_df.shape))
print('Test shape {}'.format(test_df.shape))
train_df, val_df, test_df, scaler = normalize_dfs(train_df, val_df, test_df, norm_type='standard')

if encoder_checkpoint:
    model = betaVAE(rna_features, 2048, [12000, 4096, 2048], [4096, 12000],
                      encoder_checkpoint=encoder_checkpoint)
    
    print('Restoring from checkpoint')
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded model from checkpoint')

else:
    model = betaVAE(rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=beta)
    print('Restoring from checkpoint')
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded model from checkpoint')

#if torch.cuda.is_available():
#    model = model.cuda()

model.eval()

if args.type == 'tissue':
    # Get samples from two tissues
    dataset1 = datasets['train'][0]
    dataset2 = datasets['train'][1]
    rna_columns = [x for x in train_df.columns if 'rna_' in x]

    def _get_log(x):
            # trick to take into account zeros
            x = np.log(x.replace(0, np.nan))
            return x.replace(np.nan, 0)
    # get list of columns to scale

    # log transform
    dataset1[rna_columns] = dataset1[rna_columns].apply(_get_log)
    dataset2[rna_columns] = dataset2[rna_columns].apply(_get_log)
    dataset1 = dataset1[rna_columns].values
    dataset2 = dataset2[rna_columns].values

    dataset1 = scaler.transform(dataset1)
    dataset2 = scaler.transform(dataset2)

    dataset1 = torch.from_numpy(dataset1).float()
    dataset2 = torch.from_numpy(dataset2).float()

    # get encodings from each sample
    z_mu1, z_logvar1, sample1_encod = model.encode(dataset1)
    z_mu2, z_logvar2, sample2_encod = model.encode(dataset2)

    # getting the centroids for each class
    z_mu1_centroid = z_mu1.mean(axis=0)
    z_mu2_centroid = z_mu2.mean(axis=0)

    # get difference between the encodings
    difference1 = z_mu1_centroid - z_mu2_centroid
    difference2 = z_mu2_centroid - z_mu1_centroid

    # forward pass over decoder
    recons_1 = model.decode(sample1_encod + difference1)
    recons_2 = model.decode(sample2_encod + difference2)

elif args.type == 'sex':
    dataset = datasets['train'][0]
    ref = pd.read_csv('../../GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt', sep='\t')
    wsi_file_name = dataset['wsi_file_name'].values
    wsi_male = []
    wsi_female = []
    for _, row in ref.iterrows():
        sex = row['SEX']
        for wsi in wsi_file_name:
            if row['SUBJID'] in wsi:
                if sex == 1:
                    wsi_male.append(wsi)
                else:
                    wsi_female.append(wsi)
                break

    rna_columns = [x for x in train_df.columns if 'rna_' in x]

    male_df = dataset.loc[dataset['wsi_file_name'].isin(wsi_male)]
    female_df = dataset.loc[dataset['wsi_file_name'].isin(wsi_female)]

    def _get_log(x):
            # trick to take into account zeros
            x = np.log(x.replace(0, np.nan))
            return x.replace(np.nan, 0)
    # get list of columns to scale

    # log transform
    male_df[rna_columns] = male_df[rna_columns].apply(_get_log)
    female_df[rna_columns] = female_df[rna_columns].apply(_get_log)
    male_df = male_df[rna_columns].values
    female_df = female_df[rna_columns].values

    male_df = scaler.transform(male_df)
    female_df = scaler.transform(female_df)

    male_df = torch.from_numpy(male_df).float()
    female_df = torch.from_numpy(female_df).float()

    # get encodings from each sample
    z_mu1, z_logvar1, sample1_encod = model.encode(male_df)
    z_mu2, z_logvar2, sample2_encod = model.encode(female_df)

    # getting the centroids for each class
    z_mu1_centroid = z_mu1.mean(axis=0)
    z_mu2_centroid = z_mu2.mean(axis=0)

    # get difference between the encodings
    difference1 = z_mu1_centroid - z_mu2_centroid
    difference2 = z_mu2_centroid - z_mu1_centroid

    # forward pass over decoder
    recons_1 = model.decode(sample1_encod + difference1)
    recons_2 = model.decode(sample2_encod + difference2)
else:
    print('You need to choose between tissue or sex')
    exit(1)

# save the results
results = {
    'z_mu1': z_mu1,
    'z_mu2': z_mu2,
    'z_logvar1': z_logvar1,
    'z_logvar2': z_logvar2,
    'sample1_encod': sample1_encod,
    'sample2_encod': sample2_encod,
    'sample1': male_df.detach().numpy(),
    'sample2': female_df.detach().numpy(),
    'recons_1': recons_1.detach().numpy(),
    'recons_2': recons_2.detach().numpy(),
    'z_a_to_b': difference1.detach().numpy(),
    'z_b_to_a': difference2.detach().numpy()
}

f = open(os.path.join(config['save_dir'], 'interpolation_lung_sex_results.pkl'), "wb")
pickle.dump(results, f)
f.close()