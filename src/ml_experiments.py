from genericpath import exists
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import argparse
import pickle
from read_data import PatchBagDatasetHDF5
import resnet

torch.manual_seed(99)
np.random.seed(99)
torch.cuda.manual_seed(99)

class AggregationModel(nn.Module):
    def __init__(self, resnet, resnet_dim=512, num_outputs=2, use_pretrain=False):
        super(AggregationModel, self).__init__()
        self.resnet = resnet
        self.resnet_dim = resnet_dim
        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(resnet_dim, num_outputs)
                )
        self.use_pretrain = use_pretrain
    def forward_extract(self, x):
        (batch_size, bag_size, c, h, w) = x.shape
        x = x.reshape(-1, c, h, w)
        features = self.resnet.forward_extract(x)
        features = features.view(batch_size, bag_size, self.resnet_dim)
        features = features.mean(dim=1)
        return features

    def forward(self, x):
        features = self.forward_extract(x)
        return self.fc(features)

def train(model, criterion, optimizer, dataloaders, transforms,
          save_dir='checkpoints/models/', device='cpu',
          log_interval=100, summary_writer=None, num_epochs=100, 
          scheduler=None, verbose=True):
    """ 
    Train classification/regression model.
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            criterion (torch.nn): Loss function
            optimizer (torch.optim): Optimizer
            dataloaders (dict): dict containing training and validation DataLoaders
            transforms (dict): dict containing training and validation transforms
            save_dir (str): directory to save checkpoints and models.
            device (str): device to move models and data to.
            log_interval (int): 
            summary_writer (TensorboardX): to register values into tensorboard
            num_epochs (int): number of epochs of the training
            verbose (bool): whether or not to display metrics during training
        Returns:
            train_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """

    best_epoch = 0
    best_loss = np.inf
    best_outputs = {'train': [], 'val': {}}
    loss_array = {'train': [], 'val': []}
    accuracy = {'train': [], 'val':[]}
    global_summary_step = {'train': 0, 'val': 0}

    # Creates once at the beginning of training
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        sizes = {'train': 0, 'val': 0}
        inputs_seen = {'train': 0, 'val': 0}
        running_outputs = {'train': [], 'val': []}
        running_labels = {'train': [], 'val': []}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            summary_step = global_summary_step[phase]

            output_list = []

            for batch in tqdm(dataloaders[phase]):
                wsi = batch[0]
                labels = batch[1]
                size = wsi.size(0)

                labels = labels.to(device)

                wsi = wsi.to(device)
                wsi = transforms[phase](wsi)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    # Casts operations to mixed precision
                    #with torch.cuda.amp.autocast():
                    outputs = model(wsi)
                    #output_list.append(outputs.detach().cpu().numpy().flatten())
                    # saving running outputs
                    #running_outputs[phase].append(outputs.detach().cpu().numpy())
                    #running_labels[phase].append(labels.cpu().numpy())
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # Scales the loss, and calls backward()
                        # to create scaled gradients
                        loss.backward()
                        
                        # Unscales gradients and calls
                        optimizer.step()
                        
                summary_step += 1
                running_loss += loss.item() * wsi.size(0)
                sizes[phase] += size
                inputs_seen[phase] += size
                running_corrects += torch.sum(preds == labels)

                # Emptying memory
                outputs = outputs.detach()
                loss = loss.detach()
                torch.cuda.empty_cache()

            global_summary_step[phase] = summary_step
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects / sizes[phase]
            loss_array[phase].append(epoch_loss)

            #output_list = np.concatenate(output_list, axis=0)
            print('{} Loss: {:.4f}, Acc.: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
            if phase == 'val' and epoch_loss < best_loss:

                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
                best_epoch = epoch
                #best_outputs['val'] = running_outputs['val']
                #best_outputs['train'] = running_outputs['train']
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))
    
    results = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            #'best_outputs_val': np.array(best_outputs['val']).flatten(),
            #'best_outputs_train': np.array(best_outputs['train']).flatten(),
            #'labels_val': np.array(running_labels['val']).flatten(),
            #'labels_train': np.array(running_labels['train']).flatten()
        }

    return model, results

def evaluate(model, dataloader, dataset_size, transforms, criterion,
             device='cpu', verbose=True):
    """ 
    Evaluate classification model on test set
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            dataloasder (torch.utils.data.DataLoader): dataloader with the dataset
            dataset_size (int): Size of the dataset.
            transforms (torch.nn.Sequential): Transforms to be applied to the data
            device (str): Device to move the data to. Default: cpu.
            verbose (bool): whether or not to display metrics at the end
        Returns:
            test_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """
    model.eval()

    probabilities = []
    running_acc = []
    losses = []
    all_labels = []
    all_preds = []
    for batch in tqdm(dataloader):        
        wsi = batch[0]
        labels = batch[1]

        wsi = wsi.to(device)
        wsi = transforms(wsi)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(wsi)

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs,1)
        probabilities.append(outputs.detach().to('cpu').numpy())
        losses.append(loss.detach().item())
        all_labels.append(labels.detach().to('cpu').numpy())
        all_preds.append(preds.detach().to('cpu').numpy())
    probabilities = np.concatenate(probabilities, axis=0).flatten()
    preds = np.concatenate(all_preds, axis=0).flatten()
    labels = np.concatenate(all_labels, axis=0)
    accuracy = np.sum(preds == labels) / dataset_size
    f1_scores = f1_score(labels, preds, average='weighted')
    print('Loss of the model {}; Acc. {}'.format(np.mean(losses), accuracy))

    test_results = {
        'outputs': probabilities,
        'real_labels': labels,
        'preds': preds,
        'losses': losses,
        'accuracy': accuracy,
        'f1-score': f1_scores
    }

    return test_results

class ResnetSSL(torch.nn.Module):
    def __init__(self, backbone, dim=2048, num_classes=2):
        super(ResnetSSL, self).__init__()

        self.backbone = backbone
        self.linear = torch.nn.Linear(2048, num_classes)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.linear(x)
        x = self.softmax(x)
        
        return x
    def forward_extract(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return x

class TileDataset(Dataset):
    def __init__(self, csv_file):
        self.images = csv_file['wsi_file_name'].values
        self.labels = csv_file['label'].values

        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,x):
        img = read_image(self.images[x])
        label = torch.tensor(self.labels[x], dtype=torch.long)
        return img, label
def pretrain_ml_experiment(csv_path, 
                           save_dir='pretrain_ml_experiment',
                           use_pretrain=True,
                           classes=None,
                           patch_data_path='_Patches256x256_hdf5'):
    """ Experiment using pretrained weights on the GBM vs LUAD task

    Args:
        csv_path (pandas.DataFrame): File containing the paths and labels of real data
    """
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    data = pd.read_csv(csv_path)
    data = data.loc[data.label.isin(['TCGA-LUAD', 'TCGA-GBM'])]
    classes = np.unique(data.label.values)
    batch_size = 4
    test_results_splits = {}
    
    # testting on a k-fold cv on the real data
    test_accs = []
    test_f1s = []
    
    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 99)
    sp = 0
    for split in kf.split(X=data,y=data.label.values):

        if use_pretrain:
            # Load model here
            resnet50 = torchvision.models.resnet50()
            backbone_new = nn.Sequential(*list(resnet50.children())[:-1])
            ckpt = torch.load("resnet50_simclr_rnagan.pth")
            backbone_new.load_state_dict(ckpt["resnet50_parameters"])
                
            model = ResnetSSL(backbone_new)
        else:
            model = torchvision.models.resnet50()
        
        model = model.cuda(0)
        
        optimizer = AdamW(model.parameters(), weight_decay = 0.01, lr=3e-5)
        criterion = nn.CrossEntropyLoss()
        test_df = data.iloc[split[1]]
        train_df = data.iloc[split[0]]
            
        train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=99)

        le = LabelEncoder()
        le.fit(classes)
        train_df.label = le.transform(train_df.label.values)
        val_df.label = le.transform(val_df.label.values)
        test_df.label = le.transform(test_df.label.values)

        transforms_ = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])).cuda(0)

        transforms_val = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])).cuda(0)

        transforms_both = {
            'train': transforms_,
            'val': transforms_val
        }

        
        train_dataset = TileDataset(train_df)
        val_dataset = TileDataset(val_df)
        test_dataset = TileDataset(test_df)
                    num_workers=4, pin_memory=True, 
                    shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, num_workers=4,
                    pin_memory=True, shuffle=False, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset,  
                    num_workers=4, pin_memory=True, shuffle=False, batch_size=batch_size)

        dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }
        model, _ = train(model, criterion, optimizer, dataloaders, transforms_both, 
                    save_dir=save_dir,
                    device='cuda:0',
                    num_epochs=40)
            
        test_results = evaluate(model, test_dataloader, len(test_dataset),
                                transforms_val, criterion=criterion, device='cuda:0')
        name_sp = 'split_'+str(sp)
        test_results_splits[name_sp] = test_results
        test_accs.append(test_results['accuracy'])
        test_f1s.append(test_results['f1-score'])
        sp += 1
        
    print(10*'-')
    print(test_accs)
    print(f'Test acc. {np.mean(test_accs)}+-{np.std(test_accs)}')
    print(test_f1s)
    print(f'Test f1. {np.mean(test_f1s)}+-{np.std(test_f1s)}')
    print(10*'-')
    with open(os.path.join(save_dir,'gbmvsluad_experiment_test.pkl'), 'wb') as f:
        pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='GBM vs LUAD experiment')
    parser.add_argument('--csv_path', type=str, help='CSV path with the real data')
    parser.add_argument('--save_dir', type=str, help='Directory to save results')
    parser.add_argument("--use_pretrain", help="if the pretrain experiments is carried out, using or not using pretraning",
                    action="store_true")
    args = parser.parse_args()
    
    # Arguments
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    pretrain_ml_experiment(args.csv_path, 
                           save_dir=args.save_dir,
                           use_pretrain=args.use_pretrain)
    
