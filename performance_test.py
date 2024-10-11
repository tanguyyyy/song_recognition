import alive_progress
import main
import pickle
import os
import random as rd
from time import time
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

class Test:
    def __init__(self, reference_pickle='songs.pickle', samples_folder='./samples/', samples_lenght=5, N=10):
        with open('songs.pickle', 'rb') as file:
            self.database = pickle.load(file)
        self.samples_folder = samples_folder
        self.samples_lenght = samples_lenght
        self.N = N
        self.audiofiles = os.listdir(self.samples_folder)
        self.audiofiles = [item for item in self.audiofiles if item[-4:] =='.wav']
        self.success = 0

    def run(self, display=False):
        btime = time()
        for k in range(self.N):
            test_file_path = rd.choice(self.audiofiles)
            test_sample = main.Encoding(self.samples_folder+test_file_path)
            t = rd.randint(5,int(test_sample.lenght-self.samples_lenght-5))
            test_sample.crop((t,t+self.samples_lenght))
            reference_path, reference_const = rd.choice(self.database)
            match = main.Matching(reference_const, test_sample.process()).matching()
            if match == (test_file_path == reference_path):
                self.success += 1

        delta_t = (time()-btime)/self.N
        disp_time = "{0:.2f}".format(delta_t)
        if display: print(f'Test succeeded\nNumber of tests: {self.N}\nNumber of success: {self.success}\nSuccess rate: {self.success/self.N}\nProcessing time: {disp_time}s')
        return self.success/self.N, delta_t


    def time_success(self):
        T = np.arange(0.5,12,0.5)
        success = []
        times = []
        for t in T:
            test = Test(samples_lenght=t, N=200)
            s, dt = test.run(display=False)
            success.append(s)
            times.append(dt)
        fig, ax1 = plt.subplots()
        ax1.plot(T, success)
        ax1.set_ylabel('Rate of success')
        ax1.set_xlabel('Lenght of the test samples (s)')
        ax2 = ax1.twinx()
        ax2.plot(T, times, 'orange')
        ax2.set_ylabel('Time per song (s)')
        plt.show()

class Examples:
    def __init__(self) -> None:
        pass


    def constellation_example(self):
        
        SoundFile = main.Encoding('samples\Cash Machine - Anno Domini Beats.wav')

        fig, ax = plt.subplots(figsize=(16,4))
        f, t, spec = SoundFile.spectrogram((50,60),max_freq=10000)
        plt.margins(x=0)
        t += 50
        pk = peak_local_max(spec, min_distance=20, exclude_border=False).astype('float64')
        df = 625 #1echantillon <=> 625 Hz
        dt = 749 #1seconde <=> 749 échantillons
        pkk = pk.copy()
        pkk[:,1] = pk[:,1]/dt + 50
        pkk[:,0] = pk[:,0]*df
        plt.pcolormesh(t, f, spec, shading='gouraud', cmap='viridis',norm=colors.LogNorm())
        ax.add_patch(Rectangle((55, 0), 1, 2300, fill=False))
        plt.plot(pkk[:,1], pkk[:,0], 'r.')
        ax.set_ylabel('Frequency $f$(Hz)')
        ax.set_xlabel('Time $t$(s)')
        ax.set_title('Spectrogramme avec les maximaux')
        ax.text(0.51, 0.30, "zoom à $t=55s$", transform=ax.transAxes, c='k')
        plt.show()


        f, t, spec = SoundFile.spectrogram((55,56),max_freq=3500)
        t += 55
        pk = peak_local_max(spec, min_distance=20, exclude_border=False).astype('float64')
        df = 625 #1echantillon <=> 625 Hz
        dt = 749 #1seconde <=> 749 échantillons
        pkk = pk.copy()
        pkk[:,1] = pk[:,1]/dt + 55
        pkk[:,0] = pk[:,0]*df

        anchor = pkk[-2,:]
        targets = np.array([u for u in pkk if u[1] > anchor[1]])

        fig = plt.figure(figsize=(16,4))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        ax = fig.add_subplot(1,2,1)
        ax.pcolormesh(t, f, spec, shading='gouraud', cmap='viridis',norm=colors.LogNorm())
        ax.set_ylabel('Frequency $f$(Hz)')
        ax.set_xlabel('Time $t$(s)')
        ax.plot(pkk[:,1], pkk[:,0], 'r.')
        
        ax.set_title('Maximaux conservés')


        ax = fig.add_subplot(1,2,2)
        ax.set_ylabel('Frequency $f$(Hz)')
        ax.set_xlabel('Time $t$(s)')
        ax.pcolormesh(t, f, spec, shading='gouraud', cmap='viridis',norm=colors.LogNorm())
        ax.set_xlim(55, 56)
        ax.set_ylim(0, 2250)
        ax.plot(anchor[1], anchor[0], 'b+')
        ax.plot(pkk[:,1], pkk[:,0], 'r.')
        ax.text(0.15, 0.65, "Anchor: $t_1=55,26s$", transform=ax.transAxes)
        for u in targets:
            ax.plot([anchor[1], u[1]], [anchor[0], u[0]], 'black')
        ax.set_title('Construction de la constellation');
        plt.show()


    def comparison(self, key):
        SoundFile1 = main.Encoding('samples\Jal - Edge of Water - Aakash Gandhi.wav')
        SoundFile1.crop((80,120))
        A = SoundFile1.process()

        SoundFile2 = main.Encoding('samples\Lightfoot - Aaron Lieberman.wav')
        SoundFile2.crop((80,120))
        B = SoundFile2.process()

        SoundFile3 = main.Encoding('samples\Jal - Edge of Water - Aakash Gandhi.wav')
        SoundFile3.crop((100,105))
        C = SoundFile3.process()
        AC = main.Matching(A,C)
        BC = main.Matching(B,C)

        if key == 'scatterplot':
            fig = plt.figure(figsize=(16,4))
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            ax = fig.add_subplot(1,2,1)
            ax.plot(np.array(BC.dates1)/749, np.array(BC.dates2)/749, '.k')
            ax.set_xlabel('$t_1$(s)')
            ax.set_ylabel('$t_2$(s)')
            ax.set_title('different songs')

            ax = fig.add_subplot(1,2,2)
            ax.plot(np.array(AC.dates1)/749, np.array(AC.dates2)/749, '.k')
            ax.set_xlabel('$t_1$(s)')
            ax.set_ylabel('$t_2$(s)')
            Y = [i/749 for i in [0,4000]]
            X = [i/749 for i in [15000,19000]]
            ax.plot(X, Y, 'r--')
            ax.text(0.45, 0.70, "$\Delta t = 20s$", transform=ax.transAxes, c='r')
            ax.set_title('same songs')
            plt.show()

        elif key == 'histogram':
            fig = plt.figure(figsize=(16,4))
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            ax = fig.add_subplot(1,2,1)
            ax.set_title('different songs')
            diff = np.array(BC.diff).astype('float64')
            diff /= 749
            ax.hist(diff, bins = 50, density=False, color='k')
            ax.set_ylim(0,80)
            ax.set_xlabel('$t_2-t_1$(s)')
            ax = fig.add_subplot(1,2,2)
            diff = np.array(AC.diff).astype('float64')
            diff /= 749
            ax.hist(diff, bins = 50, density=False, color='k')
            ax.set_title('same songs')
            ax.set_ylim(0,80)
            ax.set_xlabel('$t_2-t_1$(s)')
            plt.show()

        elif key == 'entier_song':
            SoundFile1 = main.Encoding('samples\Jal - Edge of Water - Aakash Gandhi.wav')
            A = SoundFile1.process()
            AC = main.Matching(A,C)
            BC = main.Matching(B,C)
            fig = plt.figure(figsize=(16,8))
            fig.subplots_adjust(hspace=0.4, wspace=0.4)

            ax = fig.add_subplot(2,2,1)
            ax.plot(np.array(BC.dates1)/749, np.array(BC.dates2)/749, '.k')
            ax.set_xlabel('$t_1$(s)')
            ax.set_ylabel('$t_2$(s)')
            ax.set_title('different songs')

            ax = fig.add_subplot(2,2,2)
            ax.plot(np.array(AC.dates1)/749, np.array(AC.dates2)/749, '.k')
            ax.set_xlabel('$t_1$(s)')
            ax.set_ylabel('$t_2$(s)')
            Y = [i/749 for i in [0,4000]]
            X = [i/749 for i in [74900,78900]]
            ax.plot(X, Y, 'r--')
            ax.text(0.40, 0.70, "$\Delta t = 100s$", transform=ax.transAxes, c='r')
            ax.set_title('same songs')
            
            ax = fig.add_subplot(2,2,3)
            diff = np.array(BC.diff).astype('float64')
            diff /= 749
            ax.hist(diff, bins = 50, density=False, color='k')
            ax.set_ylim(0,80)
            ax.set_xlabel('$t_2-t_1$(s)')
            ax = fig.add_subplot(2,2,4)
            diff = np.array(AC.diff).astype('float64')
            diff /= 749
            ax.hist(diff, bins = 50, density=False, color='k')
            ax.set_ylim(0,80)
            ax.set_xlabel('$t_2-t_1$(s)')
            plt.show()

