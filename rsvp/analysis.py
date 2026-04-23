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
from sklearn import svm
from scipy import stats


# def show_plot_frt(xs, ys, sr, save_to=None):
# 	pause_dur = 1.0 / sr

# 	plt.ion()

# 	y = []
# 	x = []

# 	# plotting the first frame
# 	# graph = plt.plot(x,y)[0]
# 	# plt.pause(pause_dur)

# 	plt.xlim(min(xs)*0.9, max(xs)*1.1)
# 	plt.ylim(min(ys)*0.9, max(ys)*1.1)

# 	if save_to:
# 		save_to.mkdir(exist_ok=True)

# 	graph = None
# 	for i, (xv, yv) in enumerate(zip(xs, ys)):
# 		print(f"{i}/{len(xs)}")
# 		x.append(xv)
# 		y.append(yv)
		
# 		if graph:
# 			graph.remove()		
# 		graph = plt.plot(x, y, color="red")[0]

# 		if save_to:
# 			plt.savefig(save_to/f"{i:03d}.png")
# 		else:
# 			plt.pause(pause_dur)
	
# 	if save_to:
# 		dramerate = sr/2048
# 		print("Framerate", dramerate)
# 		(
# 			ffmpeg
# 			.input(save_to/"*.png", pattern_type="glob", framerate=dramerate)
# 			.output(save_to/'videro.mp4')
# 			.run()
# 		)


# def spectrogram_frt(stft, f0, srate):
# 	times = librosa.times_like(f0, sr=srate)

# 	# Spectrogram
# 	D = librosa.amplitude_to_db(stft, ref=np.max)
	
	
# 	fig, ax = plt.subplots()
# 	ax.set(title='pYIN fundamental frequency estimation')
# 	img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=srate, alpha=0.5)
# 	fig.colorbar(img, ax=ax, format="%+2.f dB")
# 	ax.plot(times, f0, label='f0', color='cyan', linewidth=3)

# 	winsize=100

# 	# plt.plot(times, f0)

# 	savedir = Path("spectrogram")
# 	savedir.mkdir(exist_ok=True)

# 	plt.ion()
# 	window_dur = times[winsize] - times[0]
# 	print("wdur", window_dur)
# 	for i in tqdm(range(0, len(f0)-winsize, 1)):
# 		# removing the older graph
# 		# if graph:
# 		# 	graph.remove()
		
# 		# plotting newer graph
# 		# After past winsize, then move it 
# 		t = times[i:i+winsize]
# 		# graph = plt.plot(t, f0[max(0, i-winsize):i], color="red")[0]
# 		plt.xlim(t[0]-window_dur, t[-1]-window_dur)

# 		plt.savefig(savedir/f"{i:03d}.png")
	
# 	dramerate = srate*4/2048
# 	print(dramerate)
	
# 	(
# 		ffmpeg
# 		.input(savedir/"*.png", pattern_type="glob", framerate=dramerate)
# 		.output('spectrogram_timed.mp4')
# 		.run()
# 	)

# def vr_1d(frame_fft, fft_freqs):
# 	peaks = signal.find_peaks(frame_fft)[0]
# 	peak_indices = np.argsort(frame_fft[peaks])[::-1]
# 	stft_peak_frequencies = fft_freqs[peak_indices]
	
# 	formants = np.array(stft_peak_frequencies[:3])
# 	resonance = np.mean(np.array(formants), axis=0)
# 	return resonance, formants


def vocal_resonance(stft, stft_freqs):
	stft_peaks = [signal.find_peaks(f)[0] for f in stft.transpose()]
	# For each frame, sort peaks by highest first
	stft_peak_indices = [
		p[np.argsort(block[p])[::-1]]
		for block, p in zip(stft.transpose(), stft_peaks)
	]
	# For each frame, map those peaks to their frequencies 
	stft_peak_frequencies = [stft_freqs[p] for p in stft_peak_indices]

	formants = []
	for i in range(0, 3):
		# For each sample, map to the i-th formant frequency 
		fn = np.array([d[i] if len(d) > i else 0.0 for d in stft_peak_frequencies])
		formants.append(fn)
	
	# Compute the mean of the 3 formant frequencies 
	resonance = np.mean(np.array(formants), axis=0)
	return resonance, formants


def spectral_slope(stft, stft_freqs):
	# For each stft decomposition, fit a line to its values 
	slopes = []
	for frame in stft.transpose():
		l = linregress(stft_freqs, frame)
		slopes.append(l.slope)
	slopes = np.array(slopes)
	
	return slopes


def time_this(f):
	""" Measure the amount of time taken to exectute a function. """
	st = time.time()
	result = f()
	en = time.time()
	return result, en - st


# n_fft 512 is reccommended for speech 
def plot_progressive(xs, srate, n_fft=1024):
	stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=n_fft)

	stft0 = np.abs(librosa.stft(xs, n_fft=n_fft))
	times = librosa.times_like(stft0, sr=srate, n_fft=n_fft)

	plt.ion()

	x = np.array([])
	resonance_peaks = []
	slopes_peaks = []
	# Graph to the second lowest of these 
	segment_size = n_fft//4
	graph = None
	# Keep a running stft buffer 
	# Also resonance and weight (not smoothed!)
	stft_segments = []
	resonance_segments = []
	slope_segments =[]
	for i, o in enumerate(range(segment_size, len(xs), segment_size)):
		print(f"{i+1}/{len(xs)//segment_size+1}")
		# frame = xs[o:o+winsize]
		x = xs[o-segment_size:o]

		# frame_fft = np.abs(np.fft.fft(frame))
		# fft_freqs = np.fft.fftfreq(len(frame), 1/srate)
		# resonance, _ = vr_1d(frame_fft, fft_freqs)
		# ress.append(resonance)

		# Redundant, optimize 
		stft = np.abs(librosa.stft(x, n_fft=n_fft)) 
		stft_segments.append(stft[-1])
		
		resonance, _ = vocal_resonance(stft, stft_freqs)
		resonance_segments.append(resonance)
		resonance = np.hstack(resonance_segments)
		
		slopes = -spectral_slope(stft, stft_freqs)
		slope_segments.append(slopes)
		slopes = np.hstack(slope_segments)

		# Update iff either one has a new peak 
		resonance_peaks_new, _ = signal.find_peaks(resonance)
		slopes_peaks_new, _ = signal.find_peaks(slopes)
		if (i+1)%16==0: #len(slopes_peaks_new) != len(slopes_peaks) and len(slopes) > 7 or len(resonance_peaks_new) != len(resonance_peaks):
			print(f"New resonance peak! {len(slopes_peaks_new)}")
			slopes_peaks = slopes_peaks_new
			resonance_peaks = resonance_peaks_new

			if graph:
				graph.remove()
			# graph = plt.plot(times[:len(ress)], ress, color="red")[0]
			print(times.shape, resonance.shape)
			# graph = plt.plot(smoothing(resonance), color="red")[0]
			# graph = plt.plot(smoothing(slopes), color="red")[0]
			graph = plt.plot(smoothing(slopes), smoothing(resonance), color="red")[0]
			plt.pause(0.001)
	plt.ioff()
	plt.show()


def cache_or(path, f, overwrite=False):
	""" Loads a value from a path or computes it using a function. """
	if overwrite or not path.exists():
		data = f()
		with open(path, "w") as fp:
			json.dump(data, fp)
	else:
		with open(path, "r") as fp:
			data = json.load(fp)
	return data


def metrics_by_gender(n_fft, limit=None):
	"""
	Compute metrics for each voice clip in the `./data/f` and `./data/m` directories. 
	"""

	if limit is None:
		limit = 99999999999
	ffiles = list(Path("data/f").iterdir())[:limit//2]
	mfiles = list(Path("data/m").iterdir())[:limit//2]

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
			"paths": [str(p) for p in ffiles],
			"resonance": [v.tolist() for v in ress[:len(ffiles)]],
			"weight": [v.tolist() for v in weights[:len(ffiles)]],
		},
		"m": {
			"paths": [str(p) for p in mfiles],
			"resonance": [v.tolist() for v in ress[len(ffiles):]], 
			"weight": [v.tolist() for v in weights[len(ffiles):]],
		},
	}

	return data


def gender_scatter(n_fft, limit=512):
	""" 
	Creates a scatter plot showing mean vocal weight and resonance for 
	speakers divided by gender. 
	"""

	# Make data or load cache 
	cache_file = Path(f"clips/gender_scatter_{n_fft}_{limit}.json")
	data = cache_or(cache_file, lambda: metrics_by_gender(n_fft, limit=limit))

	for t, d in data.items():
		colour = "red" if t == "f" else "blue"
		x = [np.mean(r) for r in d["weight"]]
		y = [np.mean(w) for w in d["resonance"]]
		plt.scatter(x, y, label=t, color=colour, alpha=0.5)
	plt.xlabel("Vocal Weight")
	plt.ylabel("Vocal Resonance")
	plt.xticks(rotation=15)
	plt.tight_layout()

	# clf_x = []
	# clf_y = []
	# for g in ["f", "m"]:
	# 	for i in range(0, len(data[g]["paths"])):
	# 		clf_x.append((
	# 			np.mean(data[g]["resonance"][i]),
	# 			np.mean(data[g]["weight"][i]),
	# 		))
	# 		clf_y.append(0 if g == "f" else 1)
	# clf = svm.LinearSVC()
	# clf.fit(clf_x, clf_y)
	# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

	# plt.xlim()
	


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


def similarity(path: Path, n_fft=2048, limit=512):
	# Make data or load cache 
	cache_file = Path(f"clips/gender_scatter_{n_fft}_{limit}.json")
	data = cache_or(cache_file, lambda: metrics_by_gender(n_fft, limit=limit))

	paths = []
	resonances = []
	weights = []
	for g in ["f", "m"]:
		for i in range(0, len(data[g]["paths"])):
			weights.append(np.mean(data[g]["weight"][i]))
			resonances.append(np.mean(data[g]["resonance"][i]))
			paths.append(Path(data[g]["paths"][i]))
	
	x, srate = librosa.load(Path("clips/z_trim.wav"))	
	stft = np.abs(librosa.stft(x, n_fft=n_fft))
	stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=n_fft)
	# Average vocal resonance 
	x_resonance, _ = vocal_resonance(stft, stft_freqs)
	x_resonance = np.mean(x_resonance)
	# Average vocal weight 
	x_weight = np.mean(-spectral_slope(stft, stft_freqs))

	# Compute vocal distance with these measures 
	distances = [
		np.sqrt((x_resonance - r)**2 + (x_weight - w)**2) 
		for r, w in zip(resonances, weights)
	]
	distances_i = np.argsort(distances)
	print("This clip is closest to:")
	for i in distances_i[:5]:
		print(f"{distances[i]:.2f} - {paths[i]}")



def main():

	# n_fft = 512
	# n_fft = 1024
	n_fft = 2048

	# gender_scatter(n_fft)
	# plt.savefig("clips/gender_scatter.png")
	# plt.show()

	similarity(Path("katherine_20260128_191025.wav"))

	exit(0)

	print("Load")
	x, srate = librosa.load(Path("clips/z_trim.wav"))
	# x, srate = librosa.load(Path("clips/v.wav"))
	print("x", len(x))
	print("srate", srate)

	plot_progressive(x, srate)
	exit(0)

	stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=n_fft)

	stft = np.abs(librosa.stft(x, n_fft=n_fft))
	times = librosa.times_like(stft, sr=srate, n_fft=n_fft)

	# f0, vf, vp = librosa.pyin(
	# 	x, sr=srate, 
	# 	fmin=librosa.note_to_hz('C2'),
	# 	fmax=librosa.note_to_hz('C7'),
	# )

	resonance, formants = vocal_resonance(stft, stft_freqs)
	
	showit = False
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
	ax[3].xlabel("Vocal Weight")
	ax[3].ylabel("Vocal Resonance")
	ax[3].plot(weight[cut:-cut], resonance[cut:-cut], label="Smoothed Interpolated Peaks")
	# Clip becuase the ends are always funky
	# Use windowing in real-time maybe? 
	cut = 15
	if not showit:
		plt.figure()
		show_plot_frt(weight[cut:-cut], resonance[cut:-cut], srate)
		plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()
