"""
Create a database containing the hashcodes of the songs stored 
in the specified folder (.wav files only). 
The database is saved as a pickle file as a list of dictionaries.
Each dictionary has two keys 'song' and 'hashcodes', corresponding 
to the name of the song and to the hashcodes used as signature for 
the matching algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from main import *
import pickle
from alive_progress import alive_bar


# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    folder = './samples/'

    # 1: Load the audio files
    import os
    audiofiles = os.listdir(folder)
    audiofiles = [item for item in audiofiles if item[-4:] =='.wav']

    # 2: Set the parameters of the encoder
    # Insert your code here

    # 3: Construct the database
    database = []
    with alive_bar(len(audiofiles)) as bar:
        print('Converting...')
        for file_path in audiofiles:
            E = Encoding(folder + file_path)
            database.append((E.path_name, E.process()))
            bar()

    # 4: Save the database
    with open('songs.pickle', 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

