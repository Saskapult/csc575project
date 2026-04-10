from numpy._typing._array_like import NDArray
from scipy.signal import find_peaks, medfilt, savgol_filter
from scipy.stats import linregress
from typing import Any
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt


# It's just r1 
# More higher forments 
# Hiher amp on higher harmoics 
# Perhaps ratio of higher to lower? 
def resonance_measure(stft, stft_freqs):
	print(stft.shape)
	print(stft.transpose()[0].shape)
	peaks = np.array([find_peaks(f) for f in stft.transpose()])
	# argmax(f[peaks])
	r = np.array(stft_freqs[np.argsort(f[p])] for f, p in zip(stft, peaks))

	r1 = np.array(d[0] for d in r)

	# r1 = np.array([stft_freqs[np.argmax(f)] for f in stft.transpose()])
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
		res.append(l.slope)
	return np.array(res)


def spectral_tilt(stft, stft_freqs):
	res = []
	for frame in stft.transpose():
		total = np.sum(frame)
		i = 0
		t = 0.0
		for v in frame:
			t += v
			if t >= 0.95 * total:
				break
			else:
				i += 1
		res.append(stft_freqs[i])
	return np.array(res)



def weight_backup():
	print("Load")
	signal, srate = load_wav(Path("z.wav"))

	f0, vf, vp = librosa.pyin(
		signal, sr=srate, 
		fmin=librosa.note_to_hz('C2'),
		fmax=librosa.note_to_hz('C7'),
	)
	times = librosa.times_like(f0, sr=srate)

	stft = np.abs(librosa.stft(signal))
	stft_freqs = librosa.fft_frequencies(sr=srate)

	# Only the vocied segments 
	# vf = vp >= 0.1
	# stft = stft[:,vf]
	# times = times[vf]
	# Update: Unneeded! 

	fig, ax1 = plt.subplots()

	slopes = -spectal_slope(stft, stft_freqs)
	# slopes = medfilt(slopes, kernel_size=7)
	ax1.plot(times, slopes, label="Raw Spectral Slope")

	peaks, _ = find_peaks(slopes)
	resonance = resample_and_smooth(times, times[peaks], slopes[peaks])

	ax1.plot(times, resonance, label="Smoothed Interpolated Peaks")

	# ax2 = ax1.twinx()
	# ax2.set_ylabel("Pitch (Hz)")
	# ax2.plot(times, f0, label="F0", color="green", alpha=0.5)

	ax1.set_xlabel("time (seconds)")
	ax1.set_ylabel("Spectral Slope")
	ax1.legend()

	fig.tight_layout()
	plt.show()
	# plt.savefig("vocal_weight.png")
	# And that's a vocal weight measure! 


# Filters a sorted list (min first!)
def spacing_filter(block, dist):
	# Try filtering things close to each other? We get some overlap
	new = [block[0]]
	for v in block[1:]:
		if np.abs(v - new[-1]) >= dist:
			new.append(v)
	return np.array(new)


def resonance_checkpoint():
	# st.title("RSVP")
	print("Load")
	# Weight is /
	# Resonance is V
	signal, srate = load_wav(Path("z.wav"))

	f0, vf, vp = librosa.pyin(
		signal, sr=srate, 
		fmin=librosa.note_to_hz('C2'),
		fmax=librosa.note_to_hz('C7'),
	)
	times = librosa.times_like(f0, sr=srate)

	stft = np.abs(librosa.stft(signal))
	stft_freqs = librosa.fft_frequencies(sr=srate)

	# Only the vocied segments 
	# vf = vp >= 0.1
	# stft = stft[:,vf]
	# times = times[vf]
	# Update: Unneeded! 

	# Spectrogram
	D = librosa.amplitude_to_db(stft, ref=np.max)
	fig, ax = plt.subplots()
	img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=srate, alpha=0.5)
	# print(img)
	ax.set(title='pYIN fundamental frequency estimation')
	fig.colorbar(img, ax=ax, format="%+2.f dB")
	ax.plot(times, f0, label='f0', color='cyan', linewidth=3)


	# For each frame, get peaks 
	assert len(stft.transpose()[0]) == 1025
	peaks = [find_peaks(f)[0] for f in stft.transpose()]
	# For each frame, sort peaks by highest first
	peak_indices = [
		p[np.argsort(block[p])[::-1]]
		for block, p in zip(stft.transpose(), peaks)
	]
	# for f, p in zip(stft.transpose(), peaks):
	# 	print(f.shape)
	# 	print(p.shape)
	# For each frame, map those peaks to their frequencies 
	r = [stft_freqs[p] for p in peak_indices]

	# print(stft.transpose()[42][peaks], np.max(stft.transpose()[42][peaks]))

	# print(stft.shape, len(peaks), stft.transpose()[42].shape, peaks[42].shape, np.max(peaks[42]))

	# plt.plot(np.arange(0, len(stft.transpose()[42])), stft.transpose()[42])
	# # plt.plot(peaks[42], stft.transpose()[42][peaks[42]], "x")
	# plt.plot(peak_indices[42][:3], stft.transpose()[42][peak_indices[42][:3]], "x")
	# plt.show()
	# exit(0)

	# print(r[42])
	
	f1 = np.array([d[0] for d in r])
	f1 = medfilt(f1, kernel_size=7)
	# f1_peaks = find_peaks(f1)[0]
	# f1s = resample_and_smooth(times, times[f1_peaks], f1[f1_peaks])
	f1s = savgol_filter(f1, window_length=17, polyorder=0)
	plt.plot(times, f1s, label="F1", alpha=0.25, color="green")

	f2 = np.array([d[1] for d in r])
	f2 = medfilt(f2, kernel_size=7)
	# f2_peaks = find_peaks(f2)[0]
	# f2s = resample_and_smooth(times, times[f2_peaks], f2[f2_peaks])
	f2s = savgol_filter(f2, window_length=17, polyorder=0)
	plt.plot(times, f2s, label="F2", alpha=0.25, color="blue")

	f3 = np.array([d[2] for d in r])
	f3 = medfilt(f3, kernel_size=7)
	# f3_peaks = find_peaks(f3)[0]
	# f3s = resample_and_smooth(times, times[f3_peaks], f3[f3_peaks])
	f3s = savgol_filter(f3, window_length=17, polyorder=0)
	plt.plot(times, f3s, label="F3", alpha=0.25, color="purple")

	# Jess said to do this idk 
	f = (f1s + f2s + f3s) / 3
	plt.plot(times, f, label="F", alpha=0.75, color="teal", linewidth=3)


	slopes = -spectal_slope(stft, stft_freqs)
	# ax1.plot(times, slopes, label="Raw Spectral Slope")

	peaks, _ = find_peaks(slopes)
	resonance = resample_and_smooth(times, times[peaks], slopes[peaks])
	# ax1.plot(times, resonance, label="Smoothed Interpolated Peaks")

	# plt.plot(times, f / f0, label="F/F0", alpha=0.75, color="red", linewidth=3)

	# Average spacing 
	# s = (np.abs(f1s - f2s) + np.abs(f2s - f3s)) / 2
	# plt.plot(times, s, label="spacing", alpha=0.75, color="red", linewidth=3)
	# s_npeaks = find_peaks(-s)[0]
	# ss = resample_and_smooth(times, times[s_npeaks], s[s_npeaks])
	# plt.plot(times, ss, label="spacing", alpha=0.75, color="red", linewidth=3)


	# peaks, _ = find_peaks(r1)
	# resonance = resample_and_smooth(times, times[peaks], r1[peaks])
	# plt.plot(times, resonance, label="Smoothed Interpolated Peaks")

	# plt.xlabel("time (seconds)")
	# plt.ylabel("Resonanc")
	# plt.legend()
	plt.legend(loc='upper right')
	plt.show()
	# plt.savefig("vocal_weight.png")
	# And that's a vocal weight measure! 

	# Make clusters for all data points 
	# Compare somehow (see bayes lecture)

	# Elaborate on things are done but lacking, is MVP 
	# Especially the UI 


def main():
	print("Load")
	signal, srate = load_wav(Path("z_trim.wav"))

	print("pyin")
	f0, vf, vp = librosa.pyin(
		signal, sr=srate, 
		fmin=librosa.note_to_hz('C2'),
		fmax=librosa.note_to_hz('C7'),
	)
	times = librosa.times_like(f0, sr=srate)

	print("stft")
	stft = np.abs(librosa.stft(signal))
	stft_freqs = librosa.fft_frequencies(sr=srate)


	# For each frame, get peaks 
	assert len(stft.transpose()[0]) == 1025
	peaks = [find_peaks(f)[0] for f in stft.transpose()]
	# For each frame, sort peaks by highest first
	peak_indices = [
		p[np.argsort(block[p])[::-1]]
		for block, p in zip(stft.transpose(), peaks)
	]
	# for f, p in zip(stft.transpose(), peaks):
	# 	print(f.shape)
	# 	print(p.shape)
	# For each frame, map those peaks to their frequencies 
	r = [stft_freqs[p] for p in peak_indices]

	f1 = np.array([d[0] for d in r])
	f1 = medfilt(f1, kernel_size=7)
	# f1_peaks = find_peaks(f1)[0]
	# f1s = resample_and_smooth(times, times[f1_peaks], f1[f1_peaks])
	f1s = savgol_filter(f1, window_length=17, polyorder=0)

	f2 = np.array([d[1] for d in r])
	f2 = medfilt(f2, kernel_size=7)
	# f2_peaks = find_peaks(f2)[0]
	# f2s = resample_and_smooth(times, times[f2_peaks], f2[f2_peaks])
	f2s = savgol_filter(f2, window_length=17, polyorder=0)

	f3 = np.array([d[2] for d in r])
	f3 = medfilt(f3, kernel_size=7)
	# f3_peaks = find_peaks(f3)[0]
	# f3s = resample_and_smooth(times, times[f3_peaks], f3[f3_peaks])
	f3s = savgol_filter(f3, window_length=17, polyorder=0)

	# Jess said to do this idk 
	resonance = (f1s + f2s + f3s) / 3

	slopes = -spectal_slope(stft, stft_freqs)
	# slopes = medfilt(slopes, kernel_size=7)

	peaks, _ = find_peaks(slopes)
	weight = resample_and_smooth(times, times[peaks], slopes[peaks])

	# plt.plot(resonance)
	# plt.show()
	# plt.plot(weight)
	# plt.show()

	plt.xlabel("Vocal Weight")
	plt.ylabel("Vocal Resonance")
	# plt.plot(weight[50:-50], resonance[50:-50], label="Smoothed Interpolated Peaks")
	plt.plot(weight, resonance, label="Smoothed Interpolated Peaks")

	# plt.legend(loc='upper right')
	plt.show()
	# plt.savefig("vocal_weight.png")
	# And that's a vocal weight measure! 

	# Make clusters for all data points 
	# Compare somehow (see bayes lecture)

	# Elaborate on things are done but lacking, is MVP 
	# Especially the UI 



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

	weight = resonance_measure(stft, stft_freqs)
	resonance_x = np.linspace(0.0, max(times), len(weight))
	ax.plot(np.array(resonance_x[vf]), np.array(weight[vf]), label='resonance2', color='green', alpha=0.5, linewidth=3)

	weight = weight[vf]
	resonance_x = resonance_x[vf]

	resonance_peaks, _ = find_peaks(weight)
	weight = weight[resonance_peaks]
	resonance_x = resonance_x[resonance_peaks]
	weight = resample_and_smooth(times, resonance_x, weight)
	ax.plot(times, weight, label='resonance', color='green', alpha=1.0, linewidth=3)

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
