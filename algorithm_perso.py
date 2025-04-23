"""
Algorithm implementation
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from scipy.signal import stft, istft
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------
# Create a fingerprint for an audio file based on a set of hashes
# ----------------------------------------------------------------------------


class Encoding:

    """
    Class implementing the procedure for creating a fingerprint 
    for the audio files

    The fingerprint is created through the following steps
    - compute the spectrogram of the audio signal
    - extract local maxima of the spectrogram
    - create hashes using these maxima

    """

    def __init__(self,nperseg,noverlap,min_distance,time_window,freq_window):

        """
        Class constructor

        To Do
        -----

        Initialize in the constructor all the parameters required for
        creating the signature of the audio files. These parameters include for
        instance:
        - the window selected for computing the spectrogram
        - the size of the temporal window 
        - the size of the overlap between subsequent windows
        - etc.

        All these parameters should be kept as attributes of the class.
        """

        # Insert code here
        self.nperseg=nperseg
        self.noverlap=noverlap
        self.min_distance=min_distance
        self.time_window=time_window
        self.freq_window=freq_window
        # Window size = 128 samples, overlap of 32 samples,
        


    def process(self, fs, s):

        """

        To Do
        -----

        This function takes as input a sampled signal s and the sampling
        frequency fs and returns the fingerprint (the hashcodes) of the signal.
        The fingerprint is created through the following steps
        - spectrogram computation
        - local maxima extraction
        - hashes creation

        Implement all these operations in this function. Keep as attributes of
        the class the spectrogram, the range of frequencies, the anchors, the 
        list of hashes, etc.

        Each hash can conveniently be represented by a Python dictionary 
        containing the time associated to its anchor (key: "t") and a numpy 
        array with the difference in time between the anchor and the target, 
        the frequency of the anchor and the frequency of the target 
        (key: "hash")


        Parameters
        ----------

        fs: int
           sampling frequency [Hz]
        s: numpy array
           sampled signal
        """
      
        self.fs = fs
        self.s = s
        

        # Insert code here
        
         # Short-time Fourier transform using scipy implementation
      
         #Spectrogramme
        nperseg = self.nperseg
        noverlap = self.noverlap
        freq, times,coefs=spectrogram(s,fs,nperseg=nperseg,noverlap=noverlap)
        #self.S = np.abs(coefs)
        self.S=coefs
        self.t = times
        self.f = freq

      
      #Constellation
        pics=peak_local_max(self.S,min_distance=50,exclude_border=False)
        print(pics)
        constellation=[]
        for pic in pics:
            constellation.append([freq[pic[0]],times[pic[1]]])
        self.constellation=np.array(constellation)
        print(self.constellation)

        #Hachage
        dt=self.time_window
        df=self.freq_window
        hashes=[]
        for ancre in self.constellation:
            for i in self.constellation:
               if (0<i[1]-ancre[1]<=dt) and (abs(i[0]-ancre[0])<df):
                   v_ia = {"t":ancre[1],"hash":np.array([i[1]-ancre[1],ancre[0],i[0]])}
                   hashes.append(v_ia)
        self.hashes=hashes
        
        self.anchors = self.constellation
                  


    def display_spectrogram(self):

        """
        Display the spectrogram of the audio signal

        Parameters
        ----------
        display_anchors: boolean
           when set equal to True, the anchors are displayed on the
           spectrogram
        """
        display_anchors = False
        plt.pcolormesh(self.t, self.f/1e3, self.S, shading='gouraud')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [kHz]')
        
        if(display_anchors):
            plt.scatter(self.anchors[:, 0], self.anchors[:, 1]/1e3)
        plt.show()



# ----------------------------------------------------------------------------
# Compares two set of hashes in order to determine if two audio files match
# ----------------------------------------------------------------------------

class Matching:

    """
    Compare the hashes from two audio files to determine if these
    files match

    Attributes
    ----------

    hashes1: list of dictionaries
       hashes extracted as fingerprints for the first audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    hashes2: list of dictionaries
       hashes extracted as fingerprint for the second audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    matching: numpy array
       absolute times of the hashes that match together

    offset: numpy array
       time offsets between the matches
    """

    def __init__(self, hashes1, hashes2):

        """
        Compare the hashes from two audio files to determine if these
        files match

        Parameters
        ----------

        hashes1: list of dictionaries
           hashes extracted as fingerprint for the first audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target

        hashes2: list of dictionaries
           hashes extracted as fingerprint for the second audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target
          
        """


        self.hashes1 = hashes1
        self.hashes2 = hashes2

        times = np.array([item['t'] for item in self.hashes1])
        hashcodes = np.array([item['hash'] for item in self.hashes1])

        # Establish matches
        self.matching = []
        for hc in self.hashes2:
             t = hc['t']
             h = hc['hash'][np.newaxis, :]
             dist = np.sum(np.abs(hashcodes - h), axis=1)
             mask = (dist < 1e-6)
             if (mask != 0).any():
                 self.matching.append(np.array([times[mask][0], t]))
        self.matching = np.array(self.matching)

        # TODO: complete the implementation of the class by
        # 1. creating an array "offset" containing the time offsets of the 
        #    hashcodes that match
        # 2. implementing a criterion to decide whether or not both extracts
        #    match
        
        self.offsets = self.matching[:,0]-self.matching[:,1]
        
        self.criterion = True
        
        def groupe_erreur(offsets, erreur=0.5):
         offsets = sorted(offsets)
         groupes = []
         groupe_i = [offsets[0]]

         for num in offsets[1:]:
            if abs(num - groupe_i[-1]) <= erreur:
                  groupe_i.append(num)
            else:
                  groupes.append(groupe_i)
                  groupe_i = [num]

         groupes.append(groupe_i)
         return groupes
        
        compteur = (len(groupe_i) for groupe_i in groupe_erreur(self.offsets, erreur=0.5))
        pic_max = max(compteur)
        for pic in compteur:
           if pic_max < 10*pic :
              self.criterion = False
        
       
             
    def display_scatterplot(self):

        """
        Display through a scatterplot the times associated to the hashes
        that match
        """
    
        plt.scatter(self.matching[:, 0], self.matching[:, 1])
        plt.show()


    def display_histogram(self):

        """
        Display the offset histogram
        """
    
        plt.hist(self.offsets, bins=100, density=True)
        plt.xlabel('Offset (s)')
        plt.show()


