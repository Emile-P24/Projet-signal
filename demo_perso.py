import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from algorithm_perso import *

# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    # 1: Load the database
    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)

    # 2: Encoder
    nperseg=128
    noverlap=32
    min_distance=25
    time_window=1.
    freq_window=1500
    encoder = Encoding(nperseg=nperseg, noverlap=noverlap, 
      min_distance=min_distance,
      time_window=time_window, 
      freq_window=freq_window)
      
   
    # 3: Randomly get an extract from one of the songs of the database
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = './samples/' + song

    fs, s = read(filename)
    tstart = np.random.randint(20, 90)
    tmin = int(tstart*fs)
    duration = int(10*fs)

    # 4: Use the encoder to extract a signature from the extract
    encoder.process(fs, s[tmin:tmin + duration])
    hashes = encoder.hashes
    #encoder.display_spectrogram()

    # 5: TODO: Using the class Matching, compare the fingerprint to all the 
    # fingerprints in the database
    with open("songs.pickle", "rb") as fichier:
      titres = pickle.load(fichier)
    
    
      
    # Tracé des histogrammes de correspondance
  
    n = len(titres)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, titre in enumerate(titres):
        matcher = Matching(encoder.hashes, titre["hashcodes"])
        offsets = matcher.offsets

        axes[i].hist(offsets, bins=100)
        label = "RECONNUE" if matcher.criterion else "NON RECONNUE"
        if matcher.criterion:
          axes[i].set_title(f"{titre['song']} : {label}", fontsize=10, color = "green")
        else:
          axes[i].set_title(f"{titre['song']} : {label}", fontsize=10, color = "red")
        
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    #plt.tight_layout()
    plt.show()
  






