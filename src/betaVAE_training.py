import os
import json
import argparse
import datetime
import pickle

import numpy as np
import torch
from torch.optim import Adam, SGD, RAdam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import train_test_split
from warmup_scheduler import GradualWarmupScheduler

from model import *
from betaVAE import *
from read_data import *
from utils import *

parser = argparse.ArgumentParser(description='betaVAE training over RNA-Seq data')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--checkpoint', type=str, default=None,
        help='File with the checkpoint to start with')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--log', type=int, default=0,
        help='Use tensorboard for experiment logging')
parser.add_argument('--parallel', type=int, default=None,
        help='If data parallel wants to be used for training')

args = parser.parse_args()

#p.random.seed(args.seed)
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

test_labels = []
for id, dataset in enumerate(path_csv):
    print(dataset)
    df = pd.read_csv(dataset)
    train_df, test_df = train_test_split(df, test_size=0.2)

    train_df, val_df = train_test_split(train_df, test_size=0.2)

    #train_df, val_df, test_df, scaler = normalize_dfs(train_df, val_df, test_df, norm_type='minmax')

    datasets['train'].append(train_df)
    datasets['test'].append(test_df)
    datasets['val'].append(val_df)
    
    test_labels = test_labels + ([id] * test_df.shape[0])

if(len(datasets['train']) >=2):
    train_df = pd.concat([datasets['train'][0], datasets['train'][1]])
    val_df = pd.concat([datasets['val'][0], datasets['val'][1]])
    test_df = pd.concat([datasets['test'][0], datasets['test'][1]])
    for i in range(2, len(datasets['train'])):
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

train_dataset = RNADataset([train_df], quick=quick)
val_dataset = RNADataset([val_df])
test_dataset = RNADataset([test_df])

train_dataloader = DataLoader(train_dataset,batch_size=batch_size, 
               num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size, 
               num_workers=4, 
               shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=1, 
               num_workers=4, 
               shuffle=False)

# training

print('Finished loading dataset and creating dataloader')

print('Initializing models')


if encoder_checkpoint:
    model = betaVAE(rna_features, 2048, [12000, 4096, 2048], [4096, 12000],
                      encoder_checkpoint=encoder_checkpoint)
    if args.checkpoint is not None:
        print('Restoring from checkpoint')
        print(args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint))
        print('Loaded model from checkpoint')
    else:
        model.z_mu.apply(init_weights_uniform)
        model.decoder.apply(init_weights_uniform)
        model.z_logvar.apply(init_weights_uniform)
else:
    model = betaVAE(rna_features, 2048, [6000, 4000, 2048], [4000, 6000], beta=beta)
    if args.checkpoint is not None:
        print('Restoring from checkpoint')
        print(args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint))
        print('Loaded model from checkpoint')
    else:
        model.apply(init_weights_xavier)

#torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
print('Model initialized')

if args.parallel:
    print('Using more than one gpu...')
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model = model.cuda()

lr = config.get('lr', 3e-3)

if opt == 'RAdam':
    optimizer = RAdam(model.parameters(), weight_decay = config['weights_decay'], lr=lr)
elif opt == 'SGD':
    optimizer = SGD(model.parameters(), weight_decay = config['weights_decay'], lr=lr)
else:
    optimizer = Adam(model.parameters(), weight_decay = config['weights_decay'], lr=lr)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=125, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=1000, after_scheduler=scheduler)
# train model

if args.log:
    summary_writer = SummaryWriter(
            os.path.join(config['summary_path'],
                datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_{0}".format(args.flag)))

    summary_writer.add_text('config', str(config))
else:
    summary_writer = None

dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}
model, results = train_betaVAE(model, optimizer, dataloaders,
                               save_dir=config['save_dir'],
                               device=config['device'], 
                               log_interval=config['log_interval'],
                               summary_writer=summary_writer,
                               num_epochs=config['num_epochs'],
                               scheduler=scheduler_warmup)

results_test, predictions, real = evaluate_betaVAE(model, test_dataloader)

# Reversing the transformation
predictions = np.vstack(predictions)
real = np.vstack(real)
test_results = {'predictions': 0, 'real': 0}
test_results['predictions'] = scaler.inverse_transform(predictions)
test_results['real'] = scaler.inverse_transform(real)
test_results['test_ids'] = test_df['wsi_file_name'].values
test_results['test_labels'] = np.asarray(test_labels)
f = open(os.path.join(config['save_dir'], 'test_results.pkl'), "wb")
pickle.dump(test_results, f)
f.close()