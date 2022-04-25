import time
import copy
import gc
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error

from types_ import *
from utils import *


class RNAEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: List):
        super(RNAEncoder, self).__init__()

        self.in_channels = in_channels

        modules = [
        nn.Sequential(nn.Dropout())]
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_dims: int,
                out_dims: int):
        super(Decoder, self).__init__()

        self.in_h = nn.Linear(in_channels, hidden_dims)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(hidden_dims, out_dims)
    
    def forward(self, x):
        x = self.in_h(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        return torch.tanh(x)

class betaVAE(nn.Module):
    def __init__(self,
            in_channels: int,
            z_dim: int,
            encoder_dims: List,
            hidden_dims_decoder: List,
            beta:int = 2,
            encoder_checkpoint=None):
        super(betaVAE, self).__init__()
        self.encoder = RNAEncoder(in_channels, encoder_dims)
        # if there are weights for the encoder
        if encoder_checkpoint:
            self.encoder.load_state_dict(torch.load(encoder_checkpoint))
        self.z_mu = nn.Linear(z_dim, z_dim)
        self.z_logvar = nn.Linear(z_dim, z_dim)
        self.beta = beta
        self.training = True

        modules = []
        in_ch = z_dim
        for h_dim in hidden_dims_decoder:
            modules.append(
                    nn.Sequential(
                        nn.Linear(in_ch, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU())
                    )
            in_ch = h_dim
        modules.append(nn.Sequential(nn.Linear(in_ch, in_channels), nn.Tanh()))
        self.decoder = nn.Sequential(*modules)
        #self.decoder = Decoder(z_dim, hidden_dims_decoder[0], in_channels)
        self.z_dim = z_dim

    def reparametrize(self, z_mean, z_log_var):
        #z_log_var = torch.clip(z_log_var, min=-20, max=20)
        std = torch.exp(0.5*z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps*std
        
    def encode(self, x):
        x_encoded = self.encoder(x)
        z_mean = self.z_mu(x_encoded)
        z_log_var = self.z_logvar(x_encoded)

        return z_mean, z_log_var, x_encoded
        
    def forward(self, x):
        z_mean, z_log_var, x_encoded = self.encode(x)
        z = self.reparametrize(z_mean, z_log_var)
        
        out = self.decoder(z)

        return out, z_mean, z_log_var

    def sample(self,
               num_samples:int,
               current_device: int,
               interpolation: Tensor = None,
               alpha: float = 1.0) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        gene expression.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :param interpolation: (Tensor) Difference to move samples in the latent space
        :param alpha: (Tensor) Weight for moving in the latent space
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.z_dim)

        z = z.to(current_device)

        if interpolation is not None:
            z = z + torch.from_numpy(alpha * interpolation).float().to(current_device)
        
        samples = self.decoder(z)
        return samples
    
    def decode(self, x):
        return self.decoder(x)
    
def betaVAEloss(x, x_recons, z_mean, z_logvar, beta, kld_weight=0.005, training=True):
    recons_loss =F.mse_loss(x_recons, x)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - z_logvar.exp(), dim = 1), dim = 0)

    # applying the beta correction
    if training:
        # https://openreview.net/forum?id=Sy2fzU9gl
        total_loss = recons_loss + beta * kld_loss
    else:
        total_loss = recons_loss
    
    losses = {
        'total_loss': total_loss,
        'reconstruction_loss': recons_loss,
        'kl_loss': kld_loss
    }
    return losses


def train_betaVAE(model, optimizer, dataloader,
          save_dir='checkpoints/models/', device=None,
          log_interval=100, summary_writer=None, num_epochs=100, scheduler=None, 
          verbose=True):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = {}
    best_loss['total_loss'] = np.inf

    losses_array = {
        'train': {'total_loss': [],
                'reconstruction_loss': [],
                'kl_loss': []},
        'val': {'total_loss': [],
                'reconstruction_loss': [],
                'kl_loss': []}
        }
    
    #scaler = GradScaler()
    global_summary_step = {'train': 0, 'val': 0}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        sizes = {'train': 0, 'val': 0}
        inputs_seen = {'train': 0, 'val': 0}

        for phase in ['train', 'val']:
            if phase == 'train':
                 model.train()
                 model.training = True
            else:
                 model.eval()
                 model.training = False

            running_loss = {
                'total_loss': [],
                'reconstruction_loss': [],
                'kl_loss': []
            }

            summary_step = global_summary_step[phase]
            # for logging tensorboard
            last_running_loss = {
                'total_loss': 0.0,
                'reconstruction_loss': 0.0,
                'kl_loss': 0.0
            }
            
            for b_idx, batch in tqdm(enumerate(dataloader[phase])):

                if torch.cuda.is_available():
                    batch['rna_data'] = batch['rna_data'].cuda()

                optimizer.zero_grad(set_to_none=True)
                
                #with autocast():
                with torch.set_grad_enabled(phase=='train'):
                    outputs, z_mean, z_log_var = model(batch['rna_data'])
                    
                    losses = betaVAEloss(batch['rna_data'], outputs, z_mean, z_log_var, model.beta, training=model.training)

                if phase == 'train':
                    #scaler.scale(losses['total_loss']).backward()
                    losses['total_loss'].backward()
                    #scaler.step(optimizer)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    #scaler.update()

                summary_step += 1
                for key in ['total_loss', 'reconstruction_loss', 'kl_loss']:
                    running_loss[key].append(losses[key].detach().item())

                sizes[phase] += 1
                inputs_seen[phase] += batch['rna_data'].size(0)
                
                # Emptying memory
                del outputs, z_mean, z_log_var, losses
                torch.cuda.empty_cache()
                
                if (summary_step % log_interval == 0 and summary_writer is not None):
                    for key in ['total_loss', 'reconstruction_loss', 'kl_loss']:
                        loss_to_log = (np.mean(running_loss[key]) - last_running_loss[key])
                    
                        summary_writer.add_scalar("{}/{}".format(phase, key), loss_to_log, summary_step)

                        last_running_loss[key] = np.mean(running_loss[key])
                    
                    inputs_seen[phase] = 0.0
        
            global_summary_step[phase] = summary_step
            epoch_loss = {}
            for key in ['total_loss', 'reconstruction_loss', 'kl_loss']:
                epoch_loss[key] = np.mean(running_loss[key])
                losses_array[phase][key].append(epoch_loss[key])

            if verbose:
                print('{} Total Loss: {:.4f} | Reconstruction Loss: {:.4f} | KL Loss: {:.4f}'.format(
                        phase, epoch_loss['total_loss'], epoch_loss['reconstruction_loss'],
                        epoch_loss['kl_loss']))

            if phase == 'val' and epoch_loss['total_loss'] < best_loss['total_loss']:
                best_loss['total_loss'] = epoch_loss['total_loss']
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
                best_epoch = epoch

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))


    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

    results = {
        'best_epoch': best_epoch,
        'best_loss': best_loss
    }
    return model, results

def evaluate_betaVAE(model, dataloader, verbose=True):
    model.eval()
    model.training = False

    sizes = 0
    inputs_seen = 0

    running_loss = {
        'total_loss': [],
        'reconstruction_loss': [],
        'kl_loss': [] 
    }

    predictions = []
    real = []
    for b_idx, batch in tqdm(enumerate(dataloader)):

        if torch.cuda.is_available():
            batch['rna_data'] = batch['rna_data'].cuda()

        #with autocast():
        with torch.set_grad_enabled(False):
            outputs, z_mean, z_log_var = model(batch['rna_data'])
            
            losses = betaVAEloss(batch['rna_data'], outputs, z_mean, z_log_var, model.beta, training=model.training)

            predictions.append(outputs.detach().cpu().numpy().tolist())
            real.append(batch['rna_data'].detach().cpu().numpy().tolist())

        for key in ['total_loss', 'reconstruction_loss', 'kl_loss']:
            running_loss[key].append(losses[key].detach().item())

        sizes += batch['rna_data'].size(0)
        inputs_seen += batch['rna_data'].size(0)
    
    test_loss = {
        'total_loss': np.mean(running_loss['total_loss']),
        'reconstruction_loss': np.mean(running_loss['reconstruction_loss']),
        'kl_loss': np.mean(running_loss['kl_loss'])
    }

    if verbose:
        print('Total Loss: {:.4f} | Reconstruction Loss: {:.4f} | KL Loss: {:.4f}'.format(
                test_loss['total_loss'], test_loss['reconstruction_loss'],
                test_loss['kl_loss']))
    return test_loss, predictions, real

if __name__ == "__main__":
    rna_encoder = RNAEncoder(56200,[4056,2048])
    x = torch.rand((64,56200))
    out = rna_encoder(x)
    print(out.shape)
    
    betavae = betaVAE(56200, 2048, [4056, 2048], [4056, 56200])

    x = torch.rand((64, 56200))
    out, z_mean, z_log_var = betavae(x)
    losses = betavae.loss(x, out, z_mean, z_log_var)
    print(losses)




