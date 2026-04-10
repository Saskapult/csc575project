from numpy._typing._array_like import NDArray
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.stats import linregress
from typing import Any
from pathlib import Path
# import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt


# It's just r1 
# More higher forments 
# Hiher amp on higher harmoics 
# Perhaps ratio of higher to lower? 
def resonance_measure(stft, stft_freqs):
	r1 = np.array([stft_freqs[np.argmax(f)] for f in stft.transpose()])
	return r1


# Sum of amps of freqs above some point 
# Or spectral slope? 
def weight_measure(stft, stft_freqs, deviation_pt=2000):
	weight_freq_cut_i = np.argmin(np.abs(stft_freqs - deviation_pt))
	sums = np.array([np.sum(a[weight_freq_cut_i:]) for a in stft.transpose()])
	return sums

def resample_and_smooth(x_new, x_old, data):
	res = np.interp(x_new, x_old, data)
	smoothed = savgol_filter(res, window_length=17, polyorder=0)
	return smoothed


def plot_thingy(resonance, weight):
	# Res distance to 1024 
	# Cap median or filter those outside 
	# Weight 256 

	resonance = np.minimum(np.maximum((resonance - 256) / 512, -1.0), 1.0)
	weight = np.minimum(np.maximum((weight - 64) / 256, -1.0), 1.0)
	indices = (np.abs(resonance) < 1.0) & (np.abs(weight) < 1.0)
	resonance = resonance[indices]
	weight = weight[indices]

	# resonance = medfilt(resonance, kernel_size=7)
	# weight = medfilt(weight, kernel_size=7)

	plt.plot(resonance, label="res")
	plt.plot(weight, label="wei")
	plt.legend()
	plt.show()

	plt.plot(resonance, weight)
	plt.show()


def analysis():
	# Find r1 points of femme and masc, make overlap, try differ 
	p = Path("data/m")
	files = list(p.glob("*.wav"))
	for i, f in enumerate(files):
		print(f"{i+1}/{len(files)}")
		signal, srate = load_wav(f)
		f0, vi, vp = librosa.pyin(
			signal, sr=srate, 
			fmin=librosa.note_to_hz('C2'),
			fmax=librosa.note_to_hz('C7'),
		)
		times = librosa.times_like(f0, sr=srate)

		stft = np.abs(librosa.stft(signal))
		stft_freqs = librosa.fft_frequencies(sr=srate)

		resonance = resonance_measure(stft, stft_freqs)
		resonance_x = np.linspace(0.0, max(times), len(resonance))

		# Only voiced segments
		vi = vp >= 0.05
		resonance = resonance[vi]
		resonance_x = resonance_x[vi]

		print("Bef", np.mean(resonance))

		# Interp from peaks 
		resonance_peaks, _ = find_peaks(resonance)
		resonance = resonance[resonance_peaks]
		resonance_x = resonance_x[resonance_peaks]
		resonance = resample_and_smooth(times, resonance_x, resonance)

		print("Aft", np.mean(resonance))


def spectal_slope(stft, stft_freqs):
	res = []
	for frame in stft.transpose():
		l = linregress(stft_freqs, frame)
	return np.array(res)


def spectral_tilt():
	pass


def main():
	# analysis()
	# exit(0)

	# st.title("RSVP")
	print("Load")
	signal, srate = load_wav(Path("v.wav"))

	stft = np.abs(librosa.stft(signal))
	stft_freqs = librosa.fft_frequencies(sr=srate)



	exit(0)

	# freq = np.fft.fft(signal)

	print("F0")
	f0, vf, vp = librosa.pyin(
		signal, sr=srate, 
		fmin=librosa.note_to_hz('C2'),
		fmax=librosa.note_to_hz('C7'),
	)
	times = librosa.times_like(f0, sr=srate)
	vf = vp >= 0.05

	print("Show")

	stft = np.abs(librosa.stft(signal))
	stft_freqs = librosa.fft_frequencies(sr=srate)

	print(stft.shape, vf.shape, len(vf), stft_freqs.shape)


	D = librosa.amplitude_to_db(stft, ref=np.max)
	fig, ax = plt.subplots()
	img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=srate)
	print(img)
	ax.set(title='pYIN fundamental frequency estimation')
	fig.colorbar(img, ax=ax, format="%+2.f dB")
	ax.plot(times, f0, label='f0', color='cyan', linewidth=3)

	resonance = resonance_measure(stft, stft_freqs)
	resonance_x = np.linspace(0.0, max(times), len(resonance))
	ax.plot(np.array(resonance_x[vf]), np.array(resonance[vf]), label='resonance2', color='green', alpha=0.5, linewidth=3)

	resonance = resonance[vf]
	resonance_x = resonance_x[vf]

	resonance_peaks, _ = find_peaks(resonance)
	resonance = resonance[resonance_peaks]
	resonance_x = resonance_x[resonance_peaks]
	resonance = resample_and_smooth(times, resonance_x, resonance)
	ax.plot(times, resonance, label='resonance', color='green', alpha=1.0, linewidth=3)

	# weight = weight_measure(stft, stft_freqs)
	# weight_i = np.arange(0, len(weight))[weight >= 64.0]
	# # weight = weight * vf
	# ax.plot((np.arange(0, len(resonance))*max(times)/len(resonance))[weight_i], weight[weight_i], label='weight', color='blue', alpha=0.5, linewidth=3)

	ax.legend(loc='upper right')
	plt.show()

	# plot_thingy(resonance, weight)


# @st.cache_data
def load_wav(path: Path):
	return librosa.load(path)


if __name__ == "__main__":
	main()
