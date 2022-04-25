import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from sklearn.model_selection import KFold

from resnet import *

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
                    output_list.append(outputs.detach().cpu().numpy().flatten())
                    # saving running outputs
                    running_outputs[phase].append(outputs.detach().cpu().numpy())
                    running_labels[phase].append(labels.cpu().numpy())
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

            output_list = np.concatenate(output_list, axis=0)
            print('{} Loss: {:.4f}, Acc.: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
            if phase == 'val' and epoch_loss < best_loss:

                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
                best_epoch = epoch
                best_outputs['val'] = running_outputs['val']
                best_outputs['train'] = running_outputs['train']
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))
    
    results = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_outputs_val': np.array(best_outputs['val']).flatten(),
            'best_outputs_train': np.array(best_outputs['train']).flatten(),
            'labels_val': np.array(running_labels['val']).flatten(),
            'labels_train': np.array(running_labels['train']).flatten()
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
    print('Loss of the model {}; Acc. {}'.format(np.mean(losses), accuracy))

    test_results = {
        'outputs': probabilities,
        'losses': losses,
        'accuracy': accuracy
    }

    return test_results

class TileDataset(Dataset):
    def __init__(self, patch_data_path, csv_file):
        self.patch_data_path = patch_data_path

        self.images = csv_file['wsi_file_names'].values
        self.labels = csv_file['Labels'].values

        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,x):
        img = read_image(self.images[x])
        label = self.labels[x]
        return img, label
        
if __name__ == "__main__": 
    # Arguments
    patch_data_path = 'images_form/'
    csv_path = 'real_toy_example.csv'
    test_accs = []
    batch_size = 16
    data = pd.read_csv(csv_path)
    kf = KFold(n_splits = 5, shuffle = True, random_state = 99)
    for split in kf.split(data):
        #train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['Labels'], random_state=99)
        train_df = data.iloc[split[0]]
        test_df = data.iloc[split[1]]
        train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Labels'], random_state=99)

        train_dataset = TileDataset(patch_data_path, train_df)
        val_dataset = TileDataset(patch_data_path, val_df)
        test_dataset = TileDataset(patch_data_path, test_df)

        transforms_ = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        transforms_val = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        transforms_both = {
            'train': transforms_,
            'val': transforms_val
        }
        train_dataloader = DataLoader(train_dataset, 
                    num_workers=4, pin_memory=True, 
                    shuffle=True, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, num_workers=4,
        pin_memory=True, shuffle=False, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset,  
        num_workers=4, pin_memory=True, shuffle=False, batch_size=batch_size)
        
        model = resnet50()

        layers_to_train = [model.fc]
        for param in model.parameters():
            param.requires_grad = False
        for layer in layers_to_train:
            for n, param in layer.named_parameters():
                param.requires_grad = True

        model = model.cuda(0)
        optimizer = AdamW(model.parameters(), weight_decay = 0.01, lr=3e-3)
        criterion = nn.CrossEntropyLoss()
        dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }
        model, results = train(model, criterion, optimizer, dataloaders, transforms_both, 
                save_dir='toy_example',
                device='cuda:0',
                num_epochs=20)
        
        test_results = evaluate(model, test_dataloader, len(test_dataset),
                                transforms_val, criterion=criterion, device='cuda:0')

        test_accs.append(test_results['accuracy'])
    
    print(test_accs)
    print(f'Test acc. {np.mean(test_accs)}+-{np.std(test_accs)}')