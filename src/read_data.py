import os
import random
import pickle

import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lmdb
import lz4framed
import cv2

from types_ import *

class InvalidFileException(Exception):
    pass

class PatchBagRNADataset(Dataset):
    def __init__(self, patch_data_path: str, rna_csv_path: str, img_size:int , 
                    transforms=None, max_patch_per_wsi=400, bag_size=20,
                    quick=None, labels=False):
        self._patch_data_path = patch_data_path
        self._rna_csv_path = rna_csv_path
        self._img_size = img_size
        self.bag_size = bag_size
        self._transforms = transforms
        self._max_patch_per_wsi = max_patch_per_wsi
        self._quick = quick
        self._labels = labels
        self.data = {}
        self.index = []
        self._preprocess()

    def _preprocess(self):
        self.data, self.index = get_data_rna_bag_wsi(self._rna_csv_path,
                self._patch_data_path, self._max_patch_per_wsi, self.bag_size,
                self._quick, self._labels)

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        (WSI, i) = self.index[idx]
        imgs = []
        item = self.data[WSI].copy()
        for patch in item['images'][i:i + self.bag_size]:
            with open(patch, 'rb') as f:
                img = Image.open(f).convert('RGB')
            if self._transforms is not None:
                img = self._transforms(img)
            imgs.append(img)
        img = torch.stack(imgs,dim=0)
        item['patch_bag'] = img
        return item

def get_data_rna_bag_wsi(csv_path, patch_path:str , limit: int, bag_size: int, quick=None, labels=False):
    dataset = {}
    index = []
    if type(csv_path) == str:
        data = pd.read_csv(csv_path)
    else:
        data = csv_path
    
    if quick is not None:
        data = data.loc[data['wsi_file_name'].isin(quick)]

    for _, row in tqdm(data.iterrows()):
        WSI = row['wsi_file_name']
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32)
        label = row['Labels']
        label = torch.tensor(label, dtype=torch.int64)

        new_row = dict()
        new_row['WSI'] = WSI
        new_row['rna_data'] = rna_data
        new_row['label'] = label
    
        n_patches = sum(1 for _ in open(os.path.join(patch_path, WSI, 'loc.txt'))) - 2
        images = [os.path.join(patch_path, WSI, WSI + "_patch_{}.jpeg".format(i)) for i in
                  range(n_patches)]

        if limit is not None:
            images = images[:limit]
        new_row['images'] = images
        new_row['n_images'] = len(images)
        
        dataset[WSI] = {}
        dataset[WSI] = {w.lower(): new_row[w] for w in new_row.keys()}

        for k in range(len(images) // bag_size):
            index.append((WSI, bag_size * k))

    return dataset, index

class PatchBagDataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None, bag_size=40,
            max_patches_total=300, quick=False):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.data = {}
        self.index = []
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            WSI = row['wsi_file_name']
            n_patches = sum(1 for _ in open(os.path.join(self.patch_data_path, WSI, 'loc.txt'))) - 2
            n_patches = min(n_patches, self.max_patches_total)
            images = [os.path.join(self.patch_data_path, WSI, WSI + '_patch_{}.jpeg'.format(i)) for i in range(n_patches)]
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images)})
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, self.bag_size * k))

    def shuffle(self):
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        filenames = []
        (WSI, i) = self.index[idx]
        filenames.append(WSI)
        imgs = []
        row = self.data[WSI]
        for patch in row['images'][i:i + self.bag_size]:
            #with open(patch, 'rb') as f:
                #img = Image.open(f).convert('RGB')
            img = read_image(patch)
            imgs.append(img)
        img = torch.stack(imgs, dim=0)

        return img, filenames

class PatchDataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None,
            max_patches_total=300, quick=False, le=None):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.keys = []
        self.images = []
        self.filenames = []
        self.labels = []
        self.lmdbs_path = []
        self.le = le
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
            csv_file['patch_data_path'] = [self.patch_data_path] * csv_file.shape[0]
            csv_file['labels'] = [0] * csv_file.shape[0]
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(10)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            WSI = row['wsi_file_name']
            data_path = row['patch_data_path']
            label = np.asarray(row['labels'])
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
            label = torch.tensor(label, dtype=torch.float32)
            #label = label.flatten()
            try:
                path = os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))
                
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
           
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
                #n_patches = sum(1 for _ in open(os.path.join(data_path, WSI, 'loc.txt'))) - 2
                n_selected = min(n_patches, self.max_patches_total)
                n_patches= list(range(n_patches))
                n_patches_index = random.sample(n_patches, n_selected)
                '''
                n_patches_index = []
                for idx in n_patches_index_aux:
                    lmdb_value = lmdb_txn.get(keys[idx])
                    try:
                        img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
                    except:
                        continue

                    n_patches_index.append(idx)
                '''
            except:
                print('Error with db {}'.format(os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))))
                continue
            #self.keys.append(keys)
            #self.random_index.append(n_patches_index)

            for i in n_patches_index:
                #self.images.append(os.path.join(data_path, WSI, WSI + '_patch_{}.png'.format(i)))
                self.images.append(i)
                self.filenames.append(WSI)
                self.labels.append(label)
                self.lmdbs_path.append(path)
                self.keys.append(keys[i])

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except Exception as e:
            print(e)
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2,0,1)
     
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lmdb_connection = lmdb.open(self.lmdbs_path[idx],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        
        with lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[idx])

        image = self.decompress_and_deserialize(lmdb_value)

        if image == None:
            print(self.lmdbs_path[idx])
            #raise InvalidFileException("Invalid file found, skipping")
            return None
            #return image, self.labels[idx]

        return self.transforms(image), self.labels[idx]
        #return read_image(self.images[idx]), self.labels[idx]

class PatchRNADataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None,
            max_patches_total=300, quick=False, le=None):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.keys = []
        self.images = []
        self.filenames = []
        self.labels = []
        self.lmdbs_path = []
        self.rna_data_arrays = []
        self.le = le
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
            csv_file['patch_data_path'] = [self.patch_data_path] * csv_file.shape[0]
            csv_file['labels'] = [0] * csv_file.shape[0]
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            WSI = row['wsi_file_name']
            rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
            rna_data = torch.tensor(rna_data, dtype=torch.float32)
            data_path = row['patch_data_path']
            label = np.asarray(row['labels'])
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
            label = torch.tensor(label, dtype=torch.float32)
            #label = label.flatten()
            try:
                path = os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))
                if path == '../../Histology/BrainCortex_Patches256x256/GTEX-1E2YA-3025.svs/GTEX-1E2YA-3025.db': continue
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
           
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
                #n_patches = sum(1 for _ in open(os.path.join(data_path, WSI, 'loc.txt'))) - 2
                n_selected = min(n_patches, self.max_patches_total)
                n_patches= list(range(n_patches))
                n_patches_index = random.sample(n_patches, n_selected)
            except:
                print('Error with db {}'.format(os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))))
                continue
            #self.keys.append(keys)
            #self.random_index.append(n_patches_index)

            for i in n_patches_index:
                #self.images.append(os.path.join(data_path, WSI, WSI + '_patch_{}.png'.format(i)))
                self.images.append(i)
                self.filenames.append(WSI)
                self.labels.append(label)
                self.lmdbs_path.append(path)
                self.keys.append(keys[i])
                self.rna_data_arrays.append(rna_data)

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except:
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2,0,1)
     
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lmdb_connection = lmdb.open(self.lmdbs_path[idx],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        
        with lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[idx])

        image = self.decompress_and_deserialize(lmdb_value)
        rna_data = self.rna_data_arrays[idx]
        if image == None:
            print(self.lmdbs_path[idx])
            #raise InvalidFileException("Invalid file found, skipping")
            out = {
                'image': image,
                'rna_data': rna_data,
                'labels': self.labels[idx]
            }
        else:
            out = {
                'image': self.transforms(image),
                'rna_data': rna_data,
                'labels': self.labels[idx]
            }
        return out
        #return read_image(self.images[idx]), self.labels[idx]

class RNADataset(Dataset):
    def __init__(self, csv_path, quick=False):
        self._csv_path = csv_path
        self.data = None
        self.quick = quick
        self._preprocess()

    def _preprocess(self):
        self.data = get_data_rna(self._csv_path, self.quick)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def get_data_rna(csv_paths, quick=False):
    dataset = []
    for csv_path in csv_paths:
        if type(csv_path) == str:
            print('Working with dataset {}'.format(csv_path))
            data = pd.read_csv(csv_path)
        else:
            data = csv_path

        if quick:
            data = data.sample(10)

        for _, row in tqdm(data.iterrows()):
            rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
            rna_data = torch.tensor(rna_data, dtype=torch.float32)

            item = {'rna_data': rna_data}
            dataset.append(item)

    return dataset

'''
class PatchRNADataset(Dataset):
        def __init__(self, patch_data_path, rna_csv_path, img_size, transforms=None,
                     max_patch_per_wsi=400):
                self._patch_data_path = patch_data_path
                self._rna_csv_path = rna_csv_path
                self._img_size = img_size
                self._transforms = transforms
                self._max_patch_per_wsi = max_patch_per_wsi
                self.data = None
                self._preprocess()
        def _preprocess(self):
                 self.data = get_data_rna_wsi(self._rna_csv_path, self._patch_data_path,
                                              max_patches=self._max_patch_per_wsi)
        def __len__(self): return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx].copy()
            patch = item['patch']
            img = Image.open(patch).convert('RGB')
            if self._transforms is not None:
                img = self.transforms(img)
                item['img'] = img
            return item

         # TODO: function to permute the data, in order to not have
         # the same image, and create labels

def get_data_rna_wsi(csv_path, patch_path, max_patches=None):
        dataset = []
        data = pd.read_csv(csv_path)

        for _, row in tqdm(data.iterrows()):
                wsi = row['wsi_file_name']
                rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
                rna_data = torch.tensor(rna_data, dtype=torch.float32)

                new_row = dict()
                new_row['patch_folder'] = wsi
                new_row['rna_data'] = rna_data

                n_patches = sum(1 for _ in open(os.path.join(patch_path, wsi, 'loc.txt'))) - 2
                images = [os.path.join(patch_path, wsi, wsi + '_patch_{}.jpeg'.format(i)) for i in range(n_patches)]

                if max_patches is not None:
                        images = images[:max_patches]

                for i in images:
                        item = new_row.copy()
                        item['patch'] = os.path.join(patch_path, patch_path, i)
                        dataset.append(item)

                return dataset
'''

def normalize_dfs(train_df, val_df, test_df, labels=False, norm_type='standard'):
    def _get_log(x):
        # trick to take into account zeros
        x = np.log(x.replace(0, np.nan))
        return x.replace(np.nan, 0)
    # get list of columns to scale
    rna_columns = [x for x in train_df.columns if 'rna_' in x]
    
    
    # log transform
    train_df[rna_columns] = train_df[rna_columns].apply(_get_log)
    val_df[rna_columns] = val_df[rna_columns].apply(_get_log)
    test_df[rna_columns] = test_df[rna_columns].apply(_get_log)
    
    
    train_df = train_df[rna_columns+['wsi_file_name']]
    val_df = val_df[rna_columns+['wsi_file_name']]
    test_df = test_df[rna_columns+['wsi_file_name']]
    
    rna_values = train_df[rna_columns].values

    if norm_type == 'standard':
        scaler = StandardScaler()
    elif norm_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0,1))
    rna_values = scaler.fit_transform(rna_values)

    train_df[rna_columns] = rna_values
    test_df[rna_columns] = scaler.transform(test_df[rna_columns].values)
    val_df[rna_columns] = scaler.transform(val_df[rna_columns].values)

    return train_df, val_df, test_df, scaler