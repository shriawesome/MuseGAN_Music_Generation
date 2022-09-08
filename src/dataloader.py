import os
from tqdm import tqdm
import pypianoroll as pr
import numpy as np
from utils import helper 
from pathlib import Path

# This method is for downloading and extracting the data. 
# Parameters: config object
def data_downloader(config):
    
    data_dir = config.DATA_DIR

    lpd5_dir = data_dir + "/lpd_5_cleansed.tar.gz https://drive.google.com/uc?id=1yz0Ma-6cWTl6mhkrLnAVJ7RNzlQRypQ5"
    amg_dir = data_dir + "id_lists_amg.tar.gz https://drive.google.com/uc?id=1hp9b_g1hu_dkP4u8h46iqHeWMaUoI07R"
    lastfm_dir = data_dir + "id_lists_lastfm.tar.gz https://drive.google.com/uc?id=1mpsoxU2fU1AjKopkcQ8Q8V6wYmVPbnPO"

    os.system("mkdir "+ data_dir)
    os.system("gdown -O " + data_dir + "/lpd_5_cleansed.tar.gz https://drive.google.com/uc?id=1yz0Ma-6cWTl6mhkrLnAVJ7RNzlQRypQ5")
    os.system("gdown -O " + data_dir + "/id_lists_amg.tar.gz https://drive.google.com/uc?id=1hp9b_g1hu_dkP4u8h46iqHeWMaUoI07R")
    os.system("gdown -O " + data_dir + "/id_lists_lastfm.tar.gz https://drive.google.com/uc?id=1mpsoxU2fU1AjKopkcQ8Q8V6wYmVPbnPO")

    os.system("tar zxf "+data_dir + "/lpd_5_cleansed.tar.gz -C " + data_dir + "/")
    os.system("tar zxf "+data_dir + "/id_lists_amg.tar.gz -C " + data_dir + "/")
    os.system("tar zxf "+data_dir + "/id_lists_lastfm.tar.gz -C " + data_dir + "/")

# Method for extracting data
def data_prep(config):

    data_dir = config.DATA_DIR
    id_list = []
    dataset_root = Path(data_dir + config.LPD_PATH)
    for path in os.listdir('../data/amg/'):
        filepath = os.path.join("../data/amg/",path)
        if os.path.isfile(filepath):
            with open(filepath,'r') as f:
                id_list.extend([line.strip() for line in f])
    id_list = list(set(id_list))
    print("Total Tracks : ",len(id_list))
    return dataset_root, id_list

# Method to prepare data for training
def get_train_dataset(config, dataset_root, id_list):

    data = []
    pianoroll=[]
    beat_resolution = config.BEAT_RESOLUTION
    lowest_pitch = config.LOWEST_PITCH
    n_pitches = config.N_PITCHES
    measure_resolution = config.MEASURE_RESOLUTION
    n_measures = config.N_MEASURES
    n_samples_per_song = config.N_SAMPLES_PER_SONG

    # Iterate over all the songs in the ID list
    for msd_id in tqdm(id_list):
        # Load the data as pypianoroll.Multitrack instance
        song_dir = dataset_root / helper.msd_id_to_dirs(msd_id)
        multitrack = pr.load(song_dir / os.listdir(song_dir)[0])
        
        # Binarize the pianorolls
        multitrack.binarize()
        
        # Downsample the pianorolls (shape : n_timesteps x n_pithces)
        multitrack.set_resolution(beat_resolution)
        
    # Stack the pianoroll (shape : n_tracks x n_timesteps x n_pitches)
        pianoroll = (multitrack.stack() > 0)
        
        # Get the target pitch range only
        pianoroll = pianoroll[:,:,lowest_pitch:lowest_pitch + n_pitches]
        
        # Calculate the total measures
        n_total_measures = multitrack.get_max_length() // measure_resolution
        candidate = int(n_total_measures - n_measures)
        target_n_samples = int(min(n_total_measures // n_measures, n_samples_per_song))
        
        # Randomly select a number of phrases from the multitrack pianoroll
        for idx in np.random.choice(candidate, target_n_samples, False):
            start = idx * measure_resolution
            end = (idx + n_measures) * measure_resolution
            # skip the samples where some track(s) has too few notes
            if (pianoroll.sum(axis=(1,2))<10).any():
                continue
        # print(start,end,pianoroll[:,int(start):int(end)].shape)
            data.append(pianoroll[:,int(start):int(end)])
        
    print(len(data))

    np.random.shuffle(data)
    train_data = np.stack(data)
    print(f"Successfully collected {len(data)} samples from {len(id_list)} songs")
    
    try:
        np.save(config.DATA_DIR + "/processed_dataset/Train", train_data)
        print("Processed dataset saved")
    except:
        print("Failed to save the processed dataset")
    
    return train_data
