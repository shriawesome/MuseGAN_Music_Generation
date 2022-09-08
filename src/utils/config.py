import numpy as np

class Config():
    """
    Support class containing constant variables
    """
    N_TRACKS = 5  # number of tracks
    N_PITCHES = 72  # number of pitches
    LOWEST_PITCH = 24  # MIDI note number of the lowest pitch
    N_SAMPLES_PER_SONG = 8  # number of samples to extract from each song in the datset
    N_MEASURES = 4  # number of measures per sample
    BEAT_RESOLUTION = 4  # temporal resolution of a beat (in timestep)
    PROGRAMS = [0, 0, 25, 33, 48]  # program number for each track
    IS_DRUMS = [True, False, False, False, False]  # drum indicator for each track
    TRACK_NAMES = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']  # name of each track
    TEMPO = 100

    # Training constants
    BATCH_SIZE = 16
    LATENT_DIM = 128
    N_STEPS = 20000

    # Sampling constu
    SAMPLE_INTERVAL = 100  # interval to run the sampler (in step)
    N_SAMPLES = 4

    MEASURE_RESOLUTION = 4 * BEAT_RESOLUTION
    TEMPO_ARRAY = np.full((4 * 4 * MEASURE_RESOLUTION,1),100)

    DATA_DIR = "/home/joel/cs663/final_project/music-generation-gan/data"
    LPD_PATH = "/lpd_5/lpd_5_cleansed/"
    AMG_PATH = "/lpd/amg"
    