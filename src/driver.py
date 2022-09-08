import os
import os.path as path
import random
from pathlib import Path
import pickle
from dataloader import *
import dataloader

import numpy as np
import matplotlib.pyplot as plt
import torch
import pypianoroll
from pypianoroll import Multitrack, Track
from tqdm import tqdm
from utils.config import Config
from utils.helper import samples_to_midi
from utils.trainutil import get_dataloader, start_training
import generator
import discriminator


if __name__ == "__main__":


    config = Config()

    # Getting the data
    if not path.isdir(config.DATA_DIR):
        data_downloader(config)
    else:
        print("Reading from ", config.DATA_DIR)

    # Preparing the data
    dataset_root, id_list = data_prep(config)

    # Getting training dataset
    if not path.isdir(config.DATA_DIR + "/processed_dataset/"):
        train_data = get_train_dataset(config,dataset_root, id_list)
        print(train_data.shape)
    else:
        # Loading processed_dataset data
        train_data = np.load(config.DATA_DIR + "/processed_dataset/Train.npy")

    # Getting torch tensors
    data_loader = get_dataloader(train_data)

    # Building a model
    discriminator = discriminator.Discriminator()
    generator = generator.Generator()
    print("Number of parameters in G: {}".format(
        sum(p.numel() for p in generator.parameters() if p.requires_grad)))
    print("Number of parameters in D: {}".format(
        sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))

    # Create optimizers
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.9))

    # Prepare the inputs for the sampler, which wil run during the training
    sample_latent = torch.randn(config.N_SAMPLES, config.LATENT_DIM)

    # Transfer the neural nets and samples to GPU
    if torch.cuda.is_available():
        discriminator = discriminator.cuda()
        generator = generator.cuda()
        sample_latent = sample_latent.cuda()

    # Create an empty dictionary to sotre history samples
    history_samples = {}

    # Initialize step
    step = 0

    # Starting training
    history_samples = start_training(data_loader, generator, discriminator, d_optimizer, g_optimizer, sample_latent)

    # Saving the generated music
    with open('../Generated_Music/music.pkl','wb') as f:
        pickle.dump(history_samples,f)

    # Converting np.array to MIDI
    samples_to_midi("sample-music", 3, "Sample_Track")
