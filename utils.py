import torch
import numpy as np
import torchvision
from colorama import Fore, Back, Style
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt



def explore_array(name, value, p=3):
    if isinstance(value, np.ndarray):
        cprint(f'[{name}:[Shape: {value.shape}, min={value.min():.{p}f}, max={value.max():.{p}f}](numpy.ndarray)]', 
               'y')
    elif isinstance(value, torch.Tensor):
        cprint(f'[{name}:[Shape: {value.shape}, device={value.device}, min={value.min():.{p}f}, max={value.max():.{p}f}]](torch.Tensor())',
               'y')
    else:
        cprint('the input should be numpy.ndarray or torch.Tensor', 'r')
        
    return 


def explore(p=3, **kwargs):
    for k, v in kwargs.items():
        explore_array(k, v, p)

def load_png(p, size):
    x = Image.open(p).convert('RGB')

    # Define a transformation to resize the image and convert it to a tensor
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    x = transform(x)
    return x

def cprint(x, c):
    c_t = ""
    if c == 'r':
        c_t = Fore.RED
    elif c == 'g':
        c_t = Fore.GREEN
    elif c == 'y':
        c_t = Fore.YELLOW
    print(c_t, x)
    print(Style.RESET_ALL)

def si(x, p, to_01=False, normalize=False):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if to_01:
        torchvision.utils.save_image((x+1)/2, p, normalize=normalize)
    else:
        torchvision.utils.save_image(x, p, normalize=normalize)


def mp(p):
    # if p is like a/b/c/d.png, then only make a/b/c/
    first_dot = p.find('.')
    last_slash = p.rfind('/')
    if first_dot < last_slash:
        assert ValueError('Input path seems like a/b.c/d/g, which is not allowed :(')
    p_new = p[:last_slash] + '/'
    if not os.path.exists(p_new):
        os.makedirs(p_new)


def get_plt_color_list():
    return ['red', 'green', 'blue', 'black', 'orange', 'yellow', 'black']

    
   
def draw_bound(a, m, color):
    if a.device != 'cpu':
        a = a.cpu()
    if color == 'red':
        c = torch.ones((3, 224, 224)) * torch.tensor([1, 0, 0])[:, None, None]
    if color == 'green':
        c = torch.ones((3, 224, 224)) * torch.tensor([0, 1, 0])[:, None, None]
    
    return c * m + a * (1 - m)




def scatter_plot(x, y, p, l_x='x', l_y='y', color='b', marker='o', s=1, normalize=False, lim=None):
    if normalize:
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        x = 2 * x - 1
        y = 2 * y - 1
    plt.scatter(x, y, marker='o', color='b', s=s)
    plt.xlabel(l_x)
    plt.ylabel(l_y)
    plt.grid(True)
    if lim is not None:
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
    plt.savefig(p)
    plt.close()
    

    
def traj_plot(x, y, p, l_x='x', l_y='y', color='b', marker='o', normalize=False, lim=None, s=1, w=1):
    if normalize:
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())
        x = 2 * x - 1
        y = 2 * y - 1
    plt.plot(x, y, marker='o', color='b', linewidth=w, markersize=s)
    plt.xlabel(l_x)
    plt.ylabel(l_y)
    plt.grid(True)
    if lim is not None:
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
    plt.savefig(p)
    plt.close()
    

    
    