import torch

def memory_usage(device):
    """
    Check memory usage in pytorch
    """
    print(torch.cuda.get_device_name(device))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(device)/1024**3,1), 'GB')

def init_weights_xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_weights_uniform(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, -0.10, 0.10)
        m.bias.data.fill_(0.01)

# custom weights initialization called on netG and netD
def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

            