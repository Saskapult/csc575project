import json
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress
import ffmpeg
from tqdm import tqdm
import time


def show_plot_frt(xs, ys, sr, save_to=None):
	pause_dur = 1.0 / sr

	plt.ion()

	y = []
	x = []

	# plotting the first frame
	# graph = plt.plot(x,y)[0]
	# plt.pause(pause_dur)

	plt.xlim(min(xs)*0.9, max(xs)*1.1)
	plt.ylim(min(ys)*0.9, max(ys)*1.1)

	if save_to:
		save_to.mkdir(exist_ok=True)

	graph = None
	for i, (xv, yv) in enumerate(zip(xs, ys)):
		print(f"{i}/{len(xs)}")
		x.append(xv)
		y.append(yv)
		
		if graph:
			graph.remove()		
		graph = plt.plot(x, y, color="red")[0]

		if save_to:
			plt.savefig(save_to/f"{i:03d}.png")
		else:
			plt.pause(pause_dur)
	
	if save_to:
		dramerate = sr/2048
		print("Framerate", dramerate)
		(
			ffmpeg
			.input(save_to/"*.png", pattern_type="glob", framerate=dramerate)
			.output(save_to/'videro.mp4')
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


def vocal_resonance(stft, stft_freqs):
	stft_peaks = [signal.find_peaks(f)[0] for f in stft.transpose()]
	# For each frame, sort peaks by highest first
	stft_peak_indices = [
		p[np.argsort(block[p])[::-1]]
		for block, p in zip(stft.transpose(), stft_peaks)
	]
	# For each frame, map those peaks to their frequencies 
	stft_peak_frequencies = [stft_freqs[p] for p in stft_peak_indices]

	# # Could improve efficiency by finding first three max 
	# # Apparently argpartition can do this 
	# # stft_peaks_2 = np.argsort(stft.transpose(), axis=1)[:,::-1][:,:3]
	# stft_peaks_2 = np.array([signal.find_peaks(frame)[0] for frame in stft.transpose()])
	# stft_peak_indices_2 = np.argsort(stft.transpose()[stft_peaks_2], axis=1)[:,::-1]
	# # print(stft_peaks_2)
	# # print("new", stft_peaks_2[0])
	# # print("old", stft_peaks[0])
	# # exit(0)
	# print(stft_freqs.shape)
	# print(stft.shape)
	# print(stft_peaks_2.shape)
	# print("x", stft_peaks_2[0].shape)
	# print("y", stft.transpose()[0].shape)
	# i = 42
	# plt.plot(stft.transpose()[i])
	# # plt.plot(stft_peaks_2[i], stft.transpose()[i][stft_peaks_2[i]], marker="x", label="2")
	# plt.plot(stft_peak_indices_2[i][:3], stft.transpose()[i][stft_peak_indices_2[i][:3]], marker="x", label="2")
	# plt.plot(stft_peak_indices[i][:3], stft.transpose()[i][stft_peak_indices[i][:3]], marker="x", label="1")
	# plt.legend()
	# plt.show()
	# exit(0)

	# print("stft", stft)
	# print(f"p {stft_peaks}")
	# print(f"pi {stft_peak_indices}")
	# print("pf", stft_peak_frequencies)

	formants = []
	for i in range(0, 3):
		# For each sample, map to the i-th formant frequency 
		# print(stft_peak_frequencies)
		# print(stft_peak_frequencies)
		fn = np.array([d[i] if len(d) > i else 0.0 for d in stft_peak_frequencies])
		# fn = signal.medfilt(fn, kernel_size=7)
		# # f1_peaks = signal.find_peaks(f1)[0]
		# # f1s = resample_and_smooth(times, times[f1_peaks], f1[f1_peaks])
		# fn_smooth = signal.savgol_filter(fn, window_length=17, polyorder=0)
		formants.append(fn)
	# Jess said to do this idk 
	# print("formants", formants)
	# What if we weighted by their amplitudes? 
	resonance = np.mean(np.array(formants), axis=0)
	return resonance, formants


def spectral_slope(stft, stft_freqs):
	slopes = []
	for frame in stft.transpose():
		l = linregress(stft_freqs, frame)
		slopes.append(l.slope)
	slopes = np.array(slopes)
	
	return slopes


def smoothing(x):
	x = signal.medfilt(x, kernel_size=7)
	# x = signal.savgol_filter(x, window_length=17, polyorder=0)
	

	peaks, _ = signal.find_peaks(x)
	proms, _, _ = signal.peak_prominences(x, peaks)
	# print(peaks.shape)
	# print(proms.shape)

	# prom_threshold = np.mean(proms)
	# peaks = peaks[proms >= prom_threshold]

	# peak_threshold = np.median(x[peaks])
	# peak_threshold = np.quantile(x[peaks], 0.50)
	# peak_threshold = np.mean(x[peaks])
	# peaks = peaks[x[peaks] >= peak_threshold]

	# x = signal.medfilt(x, kernel_size=7)

	x = np.interp(np.arange(0, len(x)), np.arange(0, len(x))[peaks], x[peaks])
	x = uniform_filter1d(x, size=41)

	# Inflate for filter 
	# x = np.interp(np.arange(0, len(x)*2), np.arange(0, len(x))[peaks], x[peaks])

	# x = signal.savgol_filter(x, window_length=17, polyorder=0)

	# x = np.interp(np.arange(0, len(x)//2), np.arange(0, len(x)), x)

	# x = signal.medfilt(x, kernel_size=69)

	return x


def time_this(f):
	st = time.time()
	result = f()
	en = time.time()
	return result, en - st


# n_fft 512 is reccommended for speech 
def plot_progressive(xs, srate, n_fft=512):
	stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=n_fft)

	stft0 = np.abs(librosa.stft(xs, n_fft=n_fft))
	times = librosa.times_like(stft0, sr=srate*2, n_fft=n_fft)

	plt.ion()

	x = np.array([])
	resonance_peaks = []
	slopes_peaks = []
	# Graph to the second lowest of these 
	winsize = 2048
	graph = None
	for i, o in enumerate(range(0, len(xs), winsize)):
		print(f"{i+1}/{len(xs)//winsize+1}")
		frame = xs[o:o+winsize]
		x = np.append(x, frame)

		# Redundant, optimize 
		stft = np.abs(librosa.stft(x, n_fft=n_fft)) 
		resonance, _ = vocal_resonance(stft, stft_freqs)
		slopes = -spectral_slope(stft, stft_freqs)

		# Update iff either one has a new peak 
		resonance_peaks_new, _ = signal.find_peaks(resonance)
		slopes_peaks_new, _ = signal.find_peaks(slopes)
		if len(slopes_peaks_new) != len(slopes_peaks) and len(slopes) > 7 or len(resonance_peaks_new) != len(resonance_peaks):
			print(f"New resonance peak! {len(slopes_peaks_new)}")
			slopes_peaks = slopes_peaks_new
			resonance_peaks = resonance_peaks_new

			if graph:
				graph.remove()
			# graph = plt.plot(times[:len(resonance)], smoothing(resonance), color="red")[0]
			# graph = plt.plot(times[:len(slopes)], smoothing(slopes), color="red")[0]
			graph = plt.plot(smoothing(slopes), smoothing(resonance), color="red")[0]
			plt.pause(0.001)
	plt.ioff()
	plt.show()


def scatter(files):
	results = []
	for file in tqdm(files):
		x, srate = librosa.load(file)

		stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=512)
		stft = np.abs(librosa.stft(x, n_fft=512))

		resonance, _ = vocal_resonance(stft, stft_freqs)
		slopes = -spectral_slope(stft, stft_freqs)

		results.append((np.mean(resonance), np.mean(slopes)))


def gender_scatter(n_fft, overwrite=False):
	limit = 80
	ffiles = list(Path("data/f").iterdir())[:limit//2]
	mfiles = list(Path("data/m").iterdir())[:limit//2]
	# ffiles = [Path("data/f/Christie Nowak_34.wav")]

	# Make data or load cache 
	cache_file = Path(f"clips/anacache22_{n_fft}_{limit}.json")
	if overwrite or not cache_file.exists():
		data = {}

		ress = []
		weights = []
		for file in tqdm(ffiles + mfiles):
			print(file)
			x, srate = librosa.load(file)

			stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=n_fft)
			stft = np.abs(librosa.stft(x, n_fft=n_fft))

			resonance, _ = vocal_resonance(stft, stft_freqs)
			slopes = -spectral_slope(stft, stft_freqs)

			ress.append(resonance)
			weights.append(slopes)
		
		data = {
			"f": {
				"resonance": [v.tolist() for v in ress[:len(ffiles)]],
				"weight": [v.tolist() for v in weights[:len(ffiles)]],
			},
			"m": {
				"resonance": [v.tolist() for v in ress[len(ffiles):]], 
				"weight": [v.tolist() for v in weights[len(ffiles):]],
			},
		}

		with open(cache_file, "w") as fp:
			json.dump(data, fp)
	else:
		with open(cache_file, "r") as fp:
			data = json.load(fp)
	
	for t, d in data.items():
		colour = "red" if t == "f" else "blue"
		x = [np.mean(r) for r in d["weight"]]
		y = [np.mean(w) for w in d["resonance"]]
		plt.scatter(x, y, label=t, color=colour)
	plt.show()


def main():

	# n_fft = 512
	# n_fft = 1024
	n_fft = 2048

	# gender_scatter(n_fft)
	# exit(0)

	print("Load")
	x, srate = librosa.load(Path("clips/z_trim.wav"))
	# x, srate = librosa.load(Path("clips/v.wav"))
	print("x", len(x))
	print("srate", srate)

	# plot_progressive(x, srate)
	# exit(0)

	stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=n_fft)

	stft = np.abs(librosa.stft(x, n_fft=n_fft))
	times = librosa.times_like(stft, sr=srate, n_fft=n_fft)

	# f0, vf, vp = librosa.pyin(
	# 	x, sr=srate, 
	# 	fmin=librosa.note_to_hz('C2'),
	# 	fmax=librosa.note_to_hz('C7'),
	# )

	resonance, formants = vocal_resonance(stft, stft_freqs)
	
	showit = True
	fig, ax = plt.subplots(4)
	if showit:
		# ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
		for i, f in enumerate(formants):
			ax[0].plot(times, f, label=f"f{i+1}", linestyle="--")
		ax[0].plot(times, resonance, label="resonance")
		ax[0].title.set_text("No smooth")

		# r = np.zeros_like(resonance)
		# for i, f in enumerate(formants):
			# s = smoothing(f)
			# r += s
			# ax[1].plot(times, s, label=f"f{i+1}", linestyle="--")
		# r /= 3
		# Why is r different than resoannce?? 
		# R is better??? ah, it's smootheed! 
		# ax[1].plot(times, r, label="r")
		ax[1].plot(times, resonance, label="no smooth")
		ax[1].plot(times, smoothing(resonance), label="smooth")
		ax[1].plot(times, np.mean(np.array([smoothing(f) for f in formants]), axis=0), label="smooth fs")
		ax[1].title.set_text("Smooth")
		ax[1].legend()

	resonance = smoothing(resonance)

	slopes = -spectral_slope(stft, stft_freqs)
	if showit:
		ax[2].plot(times, slopes, label="no smooth")

		ax[2].plot(times, smoothing(slopes), label="smooth")

		ax[2].legend()
		# plt.show()
		# exit(0)

	weight = smoothing(slopes)


	print("Plot!")
	plt.xlabel("Vocal Weight")
	plt.ylabel("Vocal Resonance")
	# plt.plot(weight, resonance, label="Smoothed Interpolated Peaks")	
	# Clip becuase the ends are always funky
	# Use windowing in real-time maybe? 
	cut = 15
	if False:
		show_plot_frt(weight[cut:-cut], resonance[cut:-cut], srate)
		plt.ioff()
		plt.show()
	else:
		ax[3].plot(weight[cut:-cut], resonance[cut:-cut], label="Smoothed Interpolated Peaks")
		plt.show()


if __name__ == "__main__":
	main()
