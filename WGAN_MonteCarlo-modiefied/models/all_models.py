import torch
from models.convgann import ConvGAN
from models.dense_6inputs import Dense6inputs
from models.ttbar_gan import ttbarGAN

def all_models(opts):
    
    if opts.model == 'convNNforNist':
        return ConvGAN()
    if opts.model == "dense6inputs":
        return Dense6inputs()
    if opts.model == "ttbarGAN":
        return ttbarGAN()
