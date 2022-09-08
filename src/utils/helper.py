from copyreg import pickle
import os
import numpy as np
from utils import config
import pickle
import pypianoroll
from pypianoroll import Multitrack, Track

# Variable to read the constant values
conf = config.Config()


def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g.TRABAFJ128F42AF24E -> A/B/A/TRABAFJ128F42AF24E"""
    
    return os.path.join(msd_id[2],msd_id[3], msd_id[4], msd_id)


def samples_to_midi(inp_path, num_samples, filename):
    """
    Method to convert numpy array to MIDI file
    -------------------------
    Note : Use Pypianoroll 0.5.3
    """
    with open(inp_path, 'rb') as f:
        load_samples = pickle.load()

    samples, tracks = [], []
    temp = conf.TEMPO_ARRAY.reshape(256)
    key = len(load_samples.keys()) - 1

    for i in range(num_samples):
        sample = load_samples[key * 100].transpose(1, 0, 2, 3).reshape(conf.N_TRACKS, -1, conf.N_PITCHES)
        binarized = (sample > 0)
        m = Multitrack(beat_resolution = conf.BEAT_RESOLUTION, tempo = temp)

        for idx,(program, is_drum, track_name) in enumerate(zip(conf.PROGRAMS, conf.IS_DRUMS, conf.TRACK_NAMES)):
            pianoroll = np.pad(binarized[idx] > 0.5, ((0,0),(conf.LOWEST_PITCH, 128 - conf.LOWEST_PITCH - conf.N_PITCHES)))

            m.append_track(
                Track(name = track_name,
                      program = program,
                      is_drum = is_drum,
                      pianoroll = pianoroll
                     )
            )

        
        m.write("../Generated_Music/{}_{}.mid".format(filename, i))
        key -= 1