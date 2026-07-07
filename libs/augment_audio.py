import random
import numpy as np

class AudioAugmentor:
    def __init__(self, p=0.5):
        self.p = p

    def add_random_gain(self, waveform, min_gain=0.5, max_gain=1.2):
        gain = random.uniform(min_gain, max_gain)
        return waveform * gain

    def add_white_noise(self, waveform, min_snr=15, max_snr=30):
        snr = random.uniform(min_snr, max_snr)
        
        # Calculate power using numpy
        signal_power = np.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        
        # Generate random normal noise matching shape
        noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape).astype(np.float32)
        return waveform + noise

    def __call__(self, mix_waveform, ref_waveform):
        if random.random() > self.p:
            return mix_waveform, ref_waveform
            
        mix_waveform = self.add_random_gain(mix_waveform)
        ref_waveform = self.add_random_gain(ref_waveform)
        
        if random.random() > 0.5:
            mix_waveform = self.add_white_noise(mix_waveform)
            
        return mix_waveform, ref_waveform