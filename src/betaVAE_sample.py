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

parser = argparse.ArgumentParser(description='betaVAE generation of RNA-Seq data')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--checkpoint', type=str, default=None,
        help='File with the checkpoint to start with')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--log', type=int, default=0,
        help='Use tensorboard for experiment logging')
parser.add_argument('--parallel', type=int, default=None,
        help='If data parallel wants to be used for training')
parser.add_argument('--num_samples', type=int, default=64,
        help='Number of samples to generate.')
parser.add_argument('--interpolation', type=str, default=None,
        help='Interpolation file to move between samples')

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

if torch.cuda.is_available():
    model = model.cuda()

model.eval()

if args.interpolation:
    f = open(args.interpolation,'rb')
    interpolation = pickle.load(f)
    f.close()
    difference = interpolation['z_b_to_a']
else:
    difference = None

# Generating samples
generated_samples = model.sample(args.num_samples, 'cuda:0', interpolation=difference)
# Saving results
results = {}
# Reversing the transformation
results['generated'] = scaler.inverse_transform(generated_samples.detach().cpu().numpy())
f = open(os.path.join(config['save_dir'], 'generated_samples_interpolation_lung.pkl'), "wb")
pickle.dump(results, f)
f.close()


