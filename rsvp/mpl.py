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
	peaks = [signal.find_peaks(f)[0] for f in stft.transpose()]
	# For each frame, sort peaks by highest first
	peak_indices = [
		p[np.argsort(block[p])[::-1]]
		for block, p in zip(stft.transpose(), peaks)
	]
	# For each frame, map those peaks to their frequencies 
	r = [stft_freqs[p] for p in peak_indices]

	formants = []
	for i in range(0, 3):
		fn = np.array([d[i] for d in r])
		# fn = signal.medfilt(fn, kernel_size=7)
		# # f1_peaks = signal.find_peaks(f1)[0]
		# # f1s = resample_and_smooth(times, times[f1_peaks], f1[f1_peaks])
		# fn_smooth = signal.savgol_filter(fn, window_length=17, polyorder=0)
		formants.append(fn)
	# Jess said to do this idk 
	resonance = np.sum(np.array(formants), axis=0) / len(formants)
	return resonance, formants


def spectral_slope(stft, stft_freqs):
	slopes = []
	for frame in stft.transpose():
		l = linregress(stft_freqs, frame)
		slopes.append(l.slope)
	slopes = np.array(slopes)
	
	return slopes


def smoothing(x):
	peaks, _ = signal.find_peaks(x)

	peak_threshold = np.median(x[peaks])
	# peak_threshold = np.quantile(x[peaks], 0.50)
	# peaks = peaks[x[peaks] >= peak_threshold]

	x = signal.medfilt(x, kernel_size=7)

	x = resample_and_smooth(np.arange(0, len(x)), np.arange(0, len(x))[peaks], x[peaks])

	x = signal.savgol_filter(x, window_length=17, polyorder=0)

	return x


def time_this(f):
	st = time.time()
	result = f()
	en = time.time()
	return result, en - st


def main():
	# input_mode = input("Input from microphone (m) or clip (c): ")
	# if input_mode == "c":
	# 	x, srate = librosa.load(select_file(Path("clips")))
	# else:

	print("Load")
	x, srate = librosa.load(Path("clips/z_trim.wav"))
	print("x", len(x))
	print("srate", srate)

	stft_freqs = librosa.fft_frequencies(sr=srate)

	stft = np.abs(librosa.stft(x))
	times = librosa.times_like(stft, sr=srate)

	# f0, vf, vp = librosa.pyin(
	# 	x, sr=srate, 
	# 	fmin=librosa.note_to_hz('C2'),
	# 	fmax=librosa.note_to_hz('C7'),
	# )

	resonance, formants = vocal_resonance(stft, stft_freqs)
	
	if False:
		fig, ax = plt.subplots(2)
		# ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
		for i, f in enumerate(formants):
			ax[0].plot(times, f, label=f"f{i+1}", linestyle="--")
		ax[0].plot(times, resonance, label="resonance")
		ax[0].title.set_text("No smooth")

		r = np.zeros_like(resonance)
		for i, f in enumerate(formants):
			s = smoothing(f)
			r += s
			ax[1].plot(times, s, label=f"f{i+1}", linestyle="--")
		r /= 3
		ax[1].plot(times, smoothing(resonance), label="resonance")
		ax[1].plot(times, r, label="r")
		ax[1].title.set_text("Smooth")

		plt.legend()
		plt.show()
		exit(0)

	resonance = smoothing(resonance)

	slopes = -spectral_slope(stft, stft_freqs)
	if False:
		fig, ax = plt.subplots()
		ax.plot(times, slopes, label="no smooth")

		ax.plot(times, -smoothing(slopes), label="smooth")

		plt.legend()
		plt.show()
		exit(0)

	weight = smoothing(slopes)


	print("Plot!")
	plt.xlabel("Vocal Weight")
	plt.ylabel("Vocal Resonance")
	# plt.plot(weight, resonance, label="Smoothed Interpolated Peaks")	
	# Clip becuase the ends are always funky
	# Use windowing in real-time maybe? 
	cut = 45
	if False:
		show_plot_frt(weight[cut:-cut], resonance[cut:-cut], srate)
		plt.ioff()
		plt.show()
	else:
		plt.plot(weight[cut:-cut], resonance[cut:-cut], label="Smoothed Interpolated Peaks")
		plt.show()


if __name__ == "__main__":
	main()
