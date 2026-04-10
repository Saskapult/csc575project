from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import signal
from scipy.stats import linregress
import ffmpeg
from tqdm import tqdm
import time

def resample_and_smooth(x_new, x_old, data) -> np.ndarray:
	res = np.interp(x_new, x_old, data)
	smoothed = signal.savgol_filter(res, window_length=17, polyorder=0)
	return smoothed


def show_plot_frt(xs, ys, sr):
	pause_dur = 1.0 / sr

	plt.ion()

	y = []
	x = []

	# plotting the first frame
	# graph = plt.plot(x,y)[0]
	# plt.pause(pause_dur)

	plt.xlim(min(xs)*0.9, max(xs)*1.1)
	plt.ylim(min(ys)*0.9, max(ys)*1.1)
	plt.savefig("base_res.png")
	exit(0)

	savedir = Path("z_thing")
	savedir.mkdir(exist_ok=True)

	# the update loop
	graph = None
	for i, (xv, yv) in enumerate(zip(xs, ys)):
		print(f"{i}/{len(xs)}")
		x.append(xv)
		y.append(yv)
		
		# removing the older graph
		if graph:
			graph.remove()
		
		# plotting newer graph
		graph = plt.plot(x, y, color="red")[0]

		plt.savefig(savedir/f"{i:03d}.png")
		# plt.pause(pause_dur)
	
	dramerate = sr/2048
	print(dramerate)
	
	(
		ffmpeg
		.input(savedir/"*.png", pattern_type="glob", framerate=dramerate)
		.output('z_recreate.mp4')
		.run()
	)


def spectrogram_frt(stft, f0, srate):
	times = librosa.times_like(f0, sr=srate)

	# Spectrogram
	D = librosa.amplitude_to_db(stft, ref=np.max)
	
	
	fig, ax = plt.subplots()
	ax.set(title='pYIN fundamental frequency estimation')
	img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=srate, alpha=0.5)
	fig.colorbar(img, ax=ax, format="%+2.f dB")
	ax.plot(times, f0, label='f0', color='cyan', linewidth=3)

	winsize=100

	# plt.plot(times, f0)

	savedir = Path("spectrogram")
	savedir.mkdir(exist_ok=True)

	plt.ion()
	window_dur = times[winsize] - times[0]
	print("wdur", window_dur)
	for i in tqdm(range(0, len(f0)-winsize, 1)):
		# removing the older graph
		# if graph:
		# 	graph.remove()
		
		# plotting newer graph
		# After past winsize, then move it 
		t = times[i:i+winsize]
		# graph = plt.plot(t, f0[max(0, i-winsize):i], color="red")[0]
		plt.xlim(t[0]-window_dur, t[-1]-window_dur)

		plt.savefig(savedir/f"{i:03d}.png")
	
	dramerate = srate*4/2048
	print(dramerate)
	
	(
		ffmpeg
		.input(savedir/"*.png", pattern_type="glob", framerate=dramerate)
		.output('spectrogram_timed.mp4')
		.run()
	)



def main():
	print("Load")
	x, srate = librosa.load(Path("z_trim.wav"))
	print("x", len(x))
	print("srate", srate)

	print("pyin")
	st = time.time()
	f0, vf, vp = librosa.pyin(
		x, sr=srate, 
		fmin=librosa.note_to_hz('C2'),
		fmax=librosa.note_to_hz('C7'),
	)
	times = librosa.times_like(f0, sr=srate)
	print("f0", len(f0))
	print(time.time() - st)

	print("stft")
	st = time.time()
	stft = np.abs(librosa.stft(x))
	stft_freqs = librosa.fft_frequencies(sr=srate)
	print(time.time() - st)

	# spectrogram_frt(stft, f0, srate)
	# exit(0)


	# For each frame, get peaks 
	assert len(stft.transpose()[0]) == 1025
	peaks = [signal.find_peaks(f)[0] for f in stft.transpose()]
	# For each frame, sort peaks by highest first
	peak_indices = [
		p[np.argsort(block[p])[::-1]]
		for block, p in zip(stft.transpose(), peaks)
	]
	# For each frame, map those peaks to their frequencies 
	r = [stft_freqs[p] for p in peak_indices]

	# Compute resonance from first three forants 
	print("Resonance")
	r_st = time.time()
	formants = []
	for i in range(0, 3):
		f1 = np.array([d[i] for d in r])
		f1 = signal.medfilt(f1, kernel_size=7)
		# f1_peaks = signal.find_peaks(f1)[0]
		# f1s = resample_and_smooth(times, times[f1_peaks], f1[f1_peaks])
		f1s = signal.savgol_filter(f1, window_length=17, polyorder=0)
		formants.append(f1s)
	# Jess said to do this idk 
	resonance = np.sum(np.array(formants), axis=0) / len(formants)
	print(time.time() - r_st)

	# fig, ax = plt.subplots()
	# ax.set(title="resonance and f0 over time")
	# D = librosa.amplitude_to_db(stft, ref=np.max)
	# img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=srate, alpha=0.5)
	# fig.colorbar(img, ax=ax, format="%+2.f dB")
	# ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
	# for i, f in enumerate(formants):
	# 	ax.plot(times, f, label=f"f{i+1}", linestyle="--")
	# ax.plot(times, resonance, label="resonance")
	# plt.legend()
	# plt.show()

	print("Weight")
	w_st = time.time()
	# slopes = []
	# for frame in stft.transpose():
	# 	l = linregress(stft_freqs, frame)
	# 	slopes.append(l.slope)
	# slopes = np.array(slopes)
	# slopes = -slopes
	# plt.plot(slopes)
	# plt.show()

	S, phase = librosa.magphase(stft)
	slopes = librosa.feature.spectral_rolloff(S=S, sr=srate, roll_percent=0.65)[0]

	peaks, _ = signal.find_peaks(slopes)
	weight = resample_and_smooth(times, times[peaks], slopes[peaks])
	# plt.plot(times, weight)
	# plt.show()
	print(time.time() - w_st)


	# input("Plot?")

	print("Plot!")
	plt.xlabel("Vocal Weight")
	plt.ylabel("Vocal Resonance")
	# show_plot_frt(weight[50:-50], resonance[50:-50], srate)
	# plt.ioff()
	# plt.show()

	print("Done")
	# plt.plot(weight, resonance)

	
	plt.plot(weight[50:-50], resonance[50:-50], label="Smoothed Interpolated Peaks")
	plt.show()

	# plt.legend(loc='upper right')
	# plt.show()
	# plt.savefig("vocal_weight.png")
	# And that's a vocal weight measure! 

	# Make clusters for all data points 
	# Compare somehow (see bayes lecture)

	# Elaborate on things are done but lacking, is MVP 
	# Especially the UI 

if __name__ == "__main__":
	main()
