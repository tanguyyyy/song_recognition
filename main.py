import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
import scipy.signal
from skimage.feature import peak_local_max


from matplotlib.colors import Colormap
import matplotlib.colors as colors
from scipy.io.wavfile import read
from alive_progress import alive_bar

MIN_DISTANCE = 130        #La distance minimale pour la fonction peak_local_max
DELTA_T = 3           #La distance maximale en temps entre une cible et son ancre
DELTA_F = 5000          #La distance maximale en fréquence entre une cible et son ancre.
MAX_FREQ = 20000
EPSILON = 1
NPERSEG= 128

def show_parameters():
    print(f"{MIN_DISTANCE=}\n{DELTA_T=}\n{DELTA_F=}\n{MAX_FREQ=}\n{EPSILON=}\n{NPERSEG=}")
class Truc:
    def __init__(self) -> None:
        pass

class Encoding:
    """
    Our way to encode a soundfile. 
    """
    def __init__(self, path_name):
        """
        sr = soud rate, the number of samples in 1 second
        """
        self.path_name = path_name
        self.sr, self.data = read(path_name)
        self.samples = len(self.data)   #The number of samples in total
        self.lenght = self.samples/self.sr

    def time_crop(self, time_window=None):
        if time_window:
            t1, t2 = time_window
            x1, x2 = t1*self.sr, t2*self.sr
        else:
            t1, t2 = 0, self.lenght
            x1, x2 = 0, self.samples
        X = np.arange(t1, t2, 1/self.sr)
        dataview = self.data[x1:x2]
        return X, dataview

    def display(self, time_window=None):
        """
        Display the soudfile from t1 to t2 (in seconds)
        """
        X, data_view = self.time_crop(time_window)
        plt.plot(X, data_view)
        plt.xlabel('$t(s)$')
        if time_window: plt.title(f'Soundfile\nBeginning: {time_window[0]}s')
        plt.show()

    def spectrogram(self, time_window=None, max_freq=None):
        X, data_view = self.time_crop(time_window)
        f, t, spec = scipy.signal.spectrogram(data_view, self.sr, noverlap=32, nperseg=NPERSEG)
        if max_freq:
            freq_len = int(len(f)*max_freq/40000)
            return f[:freq_len], t, spec[:freq_len,:]
        else:
            return f, t, spec

    def display_spectrogram(self, time_window=None, max_freq=None):
        """
        time_window: a tuple (t1,t2) to show the spectrogram only in a small time window.
        For a better data vizualiation, I chose a logarithmic color representation for the spectogram.
        """
        f, t, spec = self.spectrogram(time_window=time_window, max_freq=max_freq)
        plt.pcolormesh(t, f, spec, shading='gouraud', cmap='viridis',norm=colors.LogNorm())
        plt.ylabel('Frequency $f$(Hz)')
        plt.xlabel('Time $t$(s)')
        if time_window: plt.title(f"Spectrogram\nColor scale: logarithmic\nBeginning: {time_window[0]}s")
        plt.colorbar()


    def crop(self, time_window):
        """
        This class method crop the Encoding object in place.
        """
        i_1, i_2 = int(time_window[0]*self.sr), int(time_window[1]*self.sr)
        self.data = self.data[i_1:i_2]
        self.samples = len(self.data)   #The number of samples in total
        self.lenght = self.samples/self.sr

    def process(self, max_freq = MAX_FREQ):
        """
        return a list of dictionnaries: {'t': t1, 'hash': (t2-t1, f1, f2)}
        """
        f, t, spec = self.spectrogram(max_freq = max_freq)
        dt = int(len(t) / self.lenght)  #1s <=> dt échantillons
        df = int(DELTA_F * len(f) / 40000)    #df échantillons  <=> DELTA_F en Hz
        pk = peak_local_max(spec, min_distance=MIN_DISTANCE, exclude_border=False)
        t_values = pk[:,1]
        f_values = pk[:,0]
        cells = []
        for i, anchor in enumerate(pk):
            targets = []
            for k, u in enumerate((pk[:,1] > anchor[1])*(pk[:,1] < anchor[1] + DELTA_T * dt)*(pk[:,0] )*(pk[:,0]<anchor[0]+df)*(pk[:,0]>anchor[0]-df)):
                if u: 
                    targets.append(pk[k,:])
            if targets: cells += Cell(anchor, targets).hash()
        return cells        

                
class Cell:
    """
    inputs
        anchor: numpy array of lenght 2, the coordinates of the anchors
        targets_list: a list of numpy arrays (the coordinates of the targets)

    returns
        {'t': t1, 'hash': (t2-t1, f1, f2)}

    """
    def __init__(self, anchor, target_list):
        self.anchor = anchor
        self.targets = target_list
    def hash(self):
        output = []
        for target in self.targets:
            dic = {}
            dic['t'] = self.anchor[1]
            dic['hash'] = (target[1] - self.anchor[1], self.anchor[0], target[0])
            output.append(dic)
        return output


class Matching:
    """
    The hash1 is the full soundfile, the hash2 is the sample.
    """
    def __init__(self, hash1, hash2):
        self.hash1 = hash1
        self.hash2 = hash2
        self.dates1 = []
        self.dates2 = []
        for h1 in hash1:
            for h2 in hash2:
                if abs(h1['hash'][0] - h2['hash'][0]) < EPSILON and h1['hash'][1:] == h2['hash'][1:]:
                    self.dates1.append(h1['t'])
                    self.dates2.append(h2['t'])
        self.diff = [t1-t2 for t1, t2 in zip(self.dates1, self.dates2)]

    def display_scatterplot(self):
        plt.scatter(self.dates1, self.dates2)
        plt.xlabel('time (in samples)')
        plt.ylabel('time (in samples)')
        plt.show()

    def display_histogram(self):
        plt.hist(self.diff, bins = 50, density=True)
        plt.show()
        

    def matching(self, k=10):
        hist = np.histogram(self.diff, bins = 100, density=True)[0]
        sorted = np.sort(hist)
        return sorted[-1]/sorted[-2] > k


        



"""

SoundFile1 = Encoding('samples\Jal - Edge of Water - Aakash Gandhi.wav')
#SoundFile1.crop((0,10))
A = SoundFile1.process()
#SoundFile1.display_spectrogram()

SoundFile2 = Encoding('samples\Jal - Edge of Water - Aakash Gandhi.wav')
SoundFile2.crop((100,105))
B = SoundFile2.process()


C = Matching(A,B)

C.display_scatterplot()
C.display_histogram()
print(C.matching())
"""


"""
key_A = [u['hash'] for u in A]
key_B = [u['hash'] for u in B]

#print(key_A)
#print(key_B)




#print(C.dates1)
#print(C.dates2)
"""
"""
[{'t': t1, 'hash':(t2-t1, f1, f2)}]


Pour les histogrammes, afficher en fonction de t1-t2
"""