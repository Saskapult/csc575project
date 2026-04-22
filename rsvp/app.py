import os
from pathlib import Path
import sys
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from threading import Lock
from scipy import signal
from analysis import cache_or, metrics_by_gender, spectral_slope, vocal_resonance
from pydub import AudioSegment
import simpleaudio


class VoiceApp:
	def __init__(self, srate, bound_margin=0.1, n_fft=2048):
		self.n_fft = n_fft
		self.srate = srate
		self.stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=self.n_fft)

		self.segment_size = self.n_fft//4
		self.buffer = np.array([])

		self.stft_segments = []
		self.resonance_segments = []
		self.slope_segments =[]

		self.lock = Lock()
		self.fig = plt.figure(42, figsize=(6, 8))
		self.ax = self.fig.subplots(3, height_ratios=[1,1,4])
		rmin, rmax, wmin, wmax, paths, rmeans, wmeans = calibrate(self.n_fft)
		rwid = rmax - rmin
		wwid = wmax - wmin
		self.wmin = wmin-wwid*bound_margin
		self.wmax = wmax+wwid*bound_margin
		self.rmin = rmin-rwid*bound_margin
		self.rmax = rmax+rwid*bound_margin
		self.paths = paths
		self.rmeans = rmeans
		self.wmeans = wmeans
		self.graph = None
		self.graphs = []
		self.closed = False

	def update(self, x):
		# print(f"Receive {len(x)} samples")
		if len(x) != self.segment_size:
			print("Keep buffer here pls")
			return
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
	
	def plot(self, limit=100):
		self.lock.acquire()
		resonance = np.hstack(self.resonance_segments[-limit:])
		slopes = np.hstack(self.slope_segments[-limit:])
		self.lock.release()

		if self.graph:
			self.graph.remove()
		for graph in self.graphs:
			graph.remove()
		self.graphs.clear()

		resonance = signal.medfilt(resonance, kernel_size=5)
		slopes = signal.medfilt(slopes, kernel_size=5)

		peak_d = 25

		self.graphs.append(self.ax[0].plot(resonance, color="red", label="resonance")[0])
		res_peaks = signal.find_peaks(resonance, distance=peak_d)[0]
		self.graphs.append(self.ax[0].plot(res_peaks, resonance[res_peaks], color="red", marker="x")[0])
		res_peaks_mf = resonance[res_peaks]#signal.medfilt(resonance[res_peaks], kernel_size=3)
		res_peaks_mf_i = np.interp(
			np.arange(0, len(resonance)),
			res_peaks,
			res_peaks_mf,
		) if len(res_peaks) > 0 else np.array([])
		# self.ax[0].legend(loc="upper left")
		self.ax[0].title.set_text("Resonance")
	
		self.graphs.append(self.ax[1].plot(slopes, color="green", label="weight")[0])
		wei_peaks = signal.find_peaks(slopes, distance=peak_d)[0]
		self.graphs.append(self.ax[1].plot(wei_peaks, slopes[wei_peaks], color="green", marker="x")[0])
		wei_peaks_mf = signal.medfilt(slopes[wei_peaks], kernel_size=3)
		wei_peaks_mf_i = np.interp(
			np.arange(0, len(slopes)),
			wei_peaks,
			wei_peaks_mf,
		) if len(wei_peaks) > 0 else np.array([])
		# self.ax[1].legend(loc="upper left")
		self.ax[1].title.set_text("Weight")

		to_i = min(
			res_peaks[-2] if len(res_peaks) >= 2 else 0,
			wei_peaks[-2] if len(wei_peaks) >= 2 else 0,
		)
		to_i = 999999999999999999
		# self.ax[0].axvline()

		self.graph = self.ax[2].plot(wei_peaks_mf_i[100:to_i], res_peaks_mf_i[100:to_i], color="blue")[0]
		self.ax[2].set_xlabel("Vocal Weight")
		self.ax[2].set_ylabel("Vocal Resonance")

		# self.ax[2].set_xbound(self.wmin, self.wmax)
		# self.ax[2].set_ybound(self.rmin, self.rmax)

		# Find similar vocies 
		# Compute vocal distance with these measures 
		distances = [
			np.sqrt((np.mean(resonance) - r)**2 + (np.mean(slopes) - w)**2) 
			for r, w in zip(self.rmeans, self.wmeans)
		]
		distances_i = np.argsort(distances)
		s = "\n".join([f"{distances[i]:.2f} - {self.paths[i]}" for i in distances_i[:3]])
		# self.ax[2].title.set_text(s)
		# self.ax[2].text(0, 0, s, va='top')
		# for i in distances_i[:5]:
			# print(f"{distances[i]:.2f} - {self.paths[i]}")
		# print("This clip is closest to:")
		# plt.figtext(0.5, -0.09, s, wrap=True, horizontalalignment='center', fontsize=14)
		# self.fig.suptitle(s, y=0.1)
		print("This voice is most similar to:\n", s)

		def on_close(_event):
			if not self.closed:
				print("Window closed!")
			self.closed = True

		self.fig.canvas.mpl_connect('close_event', on_close)


def run_rt(n_fft):
	srate = 22050
	app = VoiceApp(srate, n_fft=n_fft)

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


def plot_file(path, update_rate, n_fft):
	x, srate = librosa.load(path)
	app = VoiceApp(srate, n_fft=n_fft)

	# Discern how long each iteration should take if we want to operate in 
	# real-time 
	intended_duration = len(x) / srate
	n_iterations = len(x) // app.segment_size + 1
	time_per_iteration = intended_duration / n_iterations
	print(f"Each iteration takes {time_per_iteration*1000}ms")

	# Play the clip in the background
	seg = AudioSegment.from_wav(path)
	playback = simpleaudio.play_buffer(
		seg.raw_data, 
		num_channels=seg.channels, 
		bytes_per_sample=seg.sample_width, 
		sample_rate=seg.frame_rate
	)

	plt.ion()
	st = time.time()
	for i, o in enumerate(range(0, len(x), app.segment_size)):
		t_iteration_st = time.time()
		app.update(x[o:o+app.segment_size])
		if i > 16 and i % update_rate == 0:
			app.plot(limit=5000000)
		t_iteration_end = time.time()

		iteration_dur = t_iteration_end - t_iteration_st
		print(f"iteration took {iteration_dur*1000:.2f}ms ({iteration_dur/time_per_iteration*100:.2f}% of intended)")

		# Discern where the program should be at this given time. 
		# Wait for the appropreate amount of time before continuing. 
		intended_next_st = st + (i+1) * time_per_iteration
		pause_dur = max(0.0000000001, intended_next_st - t_iteration_end)
		print("Pause", pause_dur)
		plt.pause(pause_dur)

		if app.closed:
			print("Window closed! Exit!")
			break
	en = time.time()
	print(f"Played {en-st}s ({(en-st)/intended_duration*100:.2f}% of intended)")
	playback.stop()

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
		lambda: metrics_by_gender(n_fft, limit=limit),
		overwrite=overwrite,
	)

	all_paths = data["f"]["paths"] + data["m"]["paths"]

	all_resonance = np.hstack(data["f"]["resonance"] + data["m"]["resonance"])
	min_resonance = np.min(all_resonance)
	# max_resonance = np.max(np.hstack(all_resonance))
	max_resonance = 3000.0
	print(f"Resonance from {min_resonance} to {max_resonance}")
	all_resonance_means = np.array([np.mean(r) for r in data["f"]["resonance"] + data["m"]["resonance"]])

	all_weight = np.hstack(data["f"]["weight"] + data["m"]["weight"])
	min_weight = np.min(all_weight)
	max_weight = np.max(all_weight)
	print(f"Weight from {min_weight} to {max_weight}")
	all_weight_means = np.array([np.mean(w) for w in data["f"]["weight"] + data["m"]["weight"]])
	
	return min_resonance, max_resonance, min_weight, max_weight, all_paths, all_resonance_means, all_weight_means


def run_clip(file=None, update_rate=8, n_fft=2048):
	if file is None:
		file = select_file(Path("./clips"))
	else:
		file = Path(file)
	plot_file(file, update_rate, n_fft)


def main():
	n_fft = 2048
	if "RSVP_N_FFT" in os.environ:
		n_fft = int(os.getenv("RSVP_N_FFT"))
	update_rate = 8
	if "RSVP_UPDATE_RATE" in os.environ:
		update_rate = int(os.getenv("RSVP_UPDATE_RATE"))

	if sys.argv[1] == "rt":
		run_rt(n_fft=n_fft)
	elif sys.argv[1] == "clip":
		if len(sys.argv) > 2:
			run_clip(sys.argv[2], update_rate=update_rate, n_fft=n_fft)
		else:
			run_clip(update_rate=update_rate, n_fft=n_fft)
	else:
		print("Enter 'rt' for real-time mic input or 'clip' to analyze a clip")
		exit(1)
	# # if input("Interactive (i) or clip (c): ") == "i":
	# # 	if input("Microphone (m) or clip (c): ") == "m":
	# # 		print("TODO: mic clipped shit")
	# # print("Clipped stuff pls")

	# # run_rt()
	# plot_file(Path("clips/z_trim.wav"))



if __name__ == "__main__":
	main()
