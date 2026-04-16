from pathlib import Path
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from threading import Lock
from scipy import signal
from mpl import cache_or, gender_scatter, smoothing, spectral_slope, vocal_resonance


class VoiceApp:
	def __init__(self, srate, bound_margin=0.1):
		self.n_fft = 2048
		self.srate = srate
		self.stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=self.n_fft)

		self.segment_size = self.n_fft//4
		self.buffer = np.array([])

		self.stft_segments = []
		self.resonance_segments = []
		self.slope_segments =[]

		self.lock = Lock()
		self.fig = plt.figure(42)
		self.ax = self.fig.subplots(3, height_ratios=[1,1,4])
		rmin, rmax, wmin, wmax = calibrate(self.n_fft)
		rwid = rmax - rmin
		wwid = wmax - wmin
		self.wmin = wmin-wwid*bound_margin
		self.wmax = wmax+wwid*bound_margin
		self.rmin = rmin-rwid*bound_margin
		self.rmax = rmax+rwid*bound_margin
		self.graph = None
		self.graphs = []
		self.closed = False

	def update(self, x):
		# print(f"Receive {len(x)} samples")
		if len(x) != self.segment_size:
			print("Keep buffer here pls")
			# exit(1)
		# print(f"Process {len(x)} samples")

		stft = np.abs(librosa.stft(x, n_fft=self.n_fft)) 
		resonance, _ = vocal_resonance(stft, self.stft_freqs)
		slopes = -spectral_slope(stft, self.stft_freqs)

		self.lock.acquire()
		self.stft_segments += stft.tolist()
		self.resonance_segments.append(resonance)
		self.slope_segments.append(slopes)
		self.lock.release()
	
	def plot(self, limit=10):
		self.lock.acquire()
		resonance = np.hstack(self.resonance_segments[-limit:])
		slopes = np.hstack(self.slope_segments[-limit:])
		self.lock.release()

		if self.graph:
			self.graph.remove()
		for graph in self.graphs:
			graph.remove()
		self.graphs.clear()


		# Windows 
		# Averages? 
		# Convolve 
		# ?


		# resonance = np.convolve(resonance, np.ones(7), "valid")
		# slopes = np.convolve(slopes, np.ones(7), "valid")
		resonance = signal.medfilt(resonance, kernel_size=5)
		slopes = signal.medfilt(slopes, kernel_size=5)

		filtsz = 15000000000000
		peak_d = 25

		self.graphs.append(self.ax[0].plot(resonance, color="red", label="resonance")[0])
		res_peaks = signal.find_peaks(resonance, distance=peak_d)[0]
		self.graphs.append(self.ax[0].plot(res_peaks, resonance[res_peaks], color="red", marker="x")[0])
		res_peaks_mf = resonance[res_peaks]#signal.medfilt(resonance[res_peaks], kernel_size=3)
		# self.graphs.append(self.ax[0].plot(res_peaks, res_peaks_mf, color="blue", marker="x", label="smoothed")[0])
		res_peaks_mf_i = np.interp(
			np.arange(0, len(resonance)),
			res_peaks,
			res_peaks_mf,
		)
		# if len(res_peaks_mf_i) >= filtsz:
			# res_peaks_mf_i = signal.medfilt(res_peaks_mf_i, filtsz)
		# res_peaks_mf_i = np.convolve(resonance, np.ones(71), "valid") / 71
		# self.graphs.append(self.ax[0].plot(res_peaks_mf_i, color="blue", label="smoothed")[0])
		self.ax[0].legend(loc="upper left")
	
		self.graphs.append(self.ax[1].plot(slopes, color="red", label="weight")[0])
		wei_peaks = signal.find_peaks(slopes, distance=peak_d)[0]
		self.graphs.append(self.ax[1].plot(wei_peaks, slopes[wei_peaks], color="red", marker="x")[0])
		wei_peaks_mf = signal.medfilt(slopes[wei_peaks], kernel_size=3)
		# self.graphs.append(self.ax[1].plot(wei_peaks, wei_peaks_mf, color="blue", marker="x", label="smoothed")[0])
		wei_peaks_mf_i = np.interp(
			np.arange(0, len(slopes)),
			wei_peaks,
			wei_peaks_mf,
		)
		# if len(wei_peaks_mf_i) >= filtsz:
			# wei_peaks_mf_i = signal.medfilt(wei_peaks_mf_i, filtsz)
		# wei_peaks_mf_i = np.convolve(slopes, np.ones(71), "valid") / 71
		# self.graphs.append(self.ax[1].plot(wei_peaks_mf_i, color="blue", label="smoothed")[0])
		self.ax[1].legend(loc="upper left")


		# self.graphs.append(self.ax.axhline(np.mean(resonance), color="blue", label="mean"))
		# self.graphs.append(self.ax.axhline(np.median(resonance), color="green", label="median"))

		# med = np.median(resonance)
		# samples_i = np.arange(0, len(resonance))[resonance > med]
		# interp = np.interp(np.arange(0, len(resonance)), samples_i, resonance[samples_i])

		# chunk_size = 7
		# thing = np.array_split(resonance, len(resonance) // chunk_size)
		# print(thing)
		# print([len(t) for t in thing])
		# thing = thing[:-1]
		# print(thing)
		# print([len(t) for t in thing])
		# exit(0)
		# # thing_vals = np.mean(thing, axis=1)
		# thing_vals = np.mean(thing, axis=1)
		# thing_indices = np.arange(0, len(resonance), chunk_size)

		# r_means = np.max(self.resonance_segments[-limit:], axis=1)
		# r_means = signal.medfilt(r_means, kernel_size=7)
		# print("moree", len(self.resonance_segments[-limit:]))
		# print("basee", resonance.shape)
		# print("means", r_means.shape)
		# r_mean_smaples = np.hstack(r_means)
		# print("stacked", r_mean_smaples)
		# print(len(self.resonance_segments[-limit:]))
		# interp = np.interp(
		# 	np.arange(0, len(resonance)), 
		# 	np.arange(0, len(self.resonance_segments[-limit:]))*2, 
		# 	r_means,
		# )

		# print(interp)
		# self.graphs.append(self.ax.plot(
			# thing_vals, color="purple", label="interp", linestyle="--")[0])

		# self.graphs.append(self.ax.plot(slopes, color="red")[0])
		# graph = self.ax.plot(smoothing(resonance), color="red")[0]
		# graph = plt.plot(smoothing(slopes), color="red")[0]
		# self.graph = plt.plot(smoothing(slopes), smoothing(resonance), color="red")[0]
		# self.graph = self.ax.plot(np.mean(slopes), np.mean(resonance), color="red", marker="x")[0]
		# self.graph = self.ax.plot(np.max(slopes), np.max(resonance), color="red", marker="x")[0]
		# to_i = min(np.max(res_peaks), np.max(wei_peaks))
		to_i = min(
			res_peaks[-2] if len(res_peaks) >= 2 else 0,
			wei_peaks[-2] if len(wei_peaks) >= 2 else 0,
		)
		to_i = 999999999999999999
		self.ax[0].axvline()

		self.graph = self.ax[2].plot(wei_peaks_mf_i[100:to_i], res_peaks_mf_i[100:to_i], color="red")[0]
		self.ax[2].set_xlabel("Vocal Weight")
		self.ax[2].set_ylabel("Vocal Resonance")

		# plt.pause(0.001)
		# plt.show()

		# self.ax.set_xbound(self.wmin, self.wmax)
		# self.ax.set_ybound(self.rmin, self.rmax)
		# self.ax.legend(loc="upper right")

		def on_close(_event):
			if not self.closed:
				print("Window closed!")
			self.closed = True

		self.fig.canvas.mpl_connect('close_event', on_close)


def start_recording():
	srate = 22050
	app = VoiceApp(srate)

	def callback(input_data, _frame_count, _time_info, _status):
		input_data = np.frombuffer(input_data, dtype=np.float32)
		app.update(input_data)
		return (input_data, pyaudio.paContinue)

	# Create a micrphone stream 
	# This calls the callback every frames_per_buffer samples 
	# with frames_per_buffer new samples 
	p = pyaudio.PyAudio()
	stream = p.open(
		format=pyaudio.paFloat32,
		channels=1,
		rate=srate,
		input=True,
		stream_callback=callback,
		frames_per_buffer=app.segment_size,
	)
	
	# Wait for enough samples
	while len(app.stft_segments) < 10:
		time.sleep(0.15)
	
	update_time = 1/30
	run_dur = 100000.0
	plt.ion()
	for _ in range(0, int(run_dur//update_time)):
		app.plot()
		plt.pause(update_time)
		if app.closed:
			print("Window closed! Exit!")
			break

	stream.close()
	exit(0)


def plot_file(path):
	x, srate = librosa.load(path)
	app = VoiceApp(srate)

	# Discern how long each iteration should take if we want to operate in 
	# real-time 
	intended_duration = len(x) / srate
	n_iterations = len(x) // app.segment_size + 1
	time_per_iteration = intended_duration / n_iterations
	print(f"Each iteration takes {time_per_iteration*1000}ms")

	plt.ion()
	st = time.time()
	for i, o in enumerate(range(0, len(x), app.segment_size)):
		t_iteration_st = time.time()
		app.update(x[o:o+app.segment_size])
		if i > 16:
			app.plot(limit=5000000)
		t_iteration_end = time.time()

		iteration_dur = t_iteration_end - t_iteration_st
		print(f"iteration took {iteration_dur*1000:.2f}ms ({iteration_dur/time_per_iteration*100:.2f}% of intended)")

		intended_next_st = st + (i+1) * time_per_iteration
		pause_dur = max(0.0000000001, intended_next_st - t_iteration_end)
		print("Pause", pause_dur)
		plt.pause(pause_dur)

		if app.closed:
			print("Window closed! Exit!")
			break
	en = time.time()
	print(f"Played {en-st}s ({(en-st)/intended_duration*100:.2f}% of intended)")
	
	plt.ioff()
	plt.show()


def select_file(base_directory: Path) -> Path:
	""" CLI prompt to slect a file. """
	print(f"{base_directory}")
	fs = list(base_directory.iterdir())
	for i, p in enumerate(fs):
		print(f"{i}\t/{p.name}")
	i = int(input(f"Selection: "))
	selection = fs[i]
	if selection.is_dir():
		return select_file(selection)
	else:
		return selection


def calibrate(n_fft, overwrite=False, limit=40):
	calibate_cache_path = Path(f"clips/app_calibration_{n_fft}_{limit}.json")
	data = cache_or(
		calibate_cache_path, 
		lambda: gender_scatter(n_fft, limit=limit),
		overwrite=overwrite,
	)

	all_resonance = data["f"]["resonance"] + data["m"]["resonance"]
	min_resonance = np.min(np.hstack(all_resonance))
	# max_resonance = np.max(np.hstack(all_resonance))
	max_resonance = 3000.0
	print(f"Resonance from {min_resonance} to {max_resonance}")

	all_weight = data["f"]["weight"] + data["m"]["weight"]
	min_weight = np.min(np.hstack(all_weight))
	max_weight = np.max(np.hstack(all_weight))
	print(f"Weight from {min_weight} to {max_weight}")
	
	return min_resonance, max_resonance, min_weight, max_weight


def main():
	# if input("Interactive (i) or clip (c): ") == "i":
	# 	if input("Microphone (m) or clip (c): ") == "m":
	# 		print("TODO: mic clipped shit")
	# print("Clipped stuff pls")

	# start_recording()
	plot_file(Path("clips/z_trim.wav"))



if __name__ == "__main__":
	main()
