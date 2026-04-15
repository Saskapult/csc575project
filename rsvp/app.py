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
		self.n_fft = 1024
		self.srate = srate
		self.stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=self.n_fft)

		self.segment_size = self.n_fft//4
		self.buffer = np.array([])

		self.stft_segments = []
		self.resonance_segments = []
		self.slope_segments =[]

		self.lock = Lock()
		self.fig = plt.figure(42)
		self.ax = self.fig.add_subplot()
		rmin, rmax, wmin, wmax = calibrate(self.n_fft)
		rwid = rmax - rmin
		wwid = wmax - wmin
		self.wmin = wmin-wwid*bound_margin
		self.wmax = wmax+wwid*bound_margin
		self.rmin = rmin-rwid*bound_margin
		self.rmax = rmax+rwid*bound_margin
		self.graph = None
		self.closed = False

	def update(self, x):
		# print(f"Receive {len(x)} samples")
		if len(x) != self.segment_size:
			print("Keep buffer here pls")
			exit(0)
		# print(f"Process {len(x)} samples")

		stft = np.abs(librosa.stft(x, n_fft=self.n_fft)) 
		resonance, _ = vocal_resonance(stft, self.stft_freqs)
		slopes = -spectral_slope(stft, self.stft_freqs)

		self.lock.acquire()
		self.stft_segments.append(stft[-1])
		# self.stft_segments += stft
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

		# graph = plt.plot(times[:len(ress)], ress, color="red")[0]
		# print(times.shape, resonance.shape)
		# graph = plt.plot(smoothing(resonance), color="red")[0]
		# graph = plt.plot(smoothing(slopes), color="red")[0]
		# self.graph = plt.plot(smoothing(slopes), smoothing(resonance), color="red")[0]
		self.graph = self.ax.plot(slopes, resonance, color="red")[0]
		# plt.pause(0.001)
		# plt.show()

		self.ax.set_xbound(self.wmin, self.wmax)
		self.ax.set_ybound(self.rmin, self.rmax)

		def on_close(_event):
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
	for i, o in enumerate(range(0, len(x), app.segment_size)):
		print(i)

		t_iteration_st = time.time()
		app.update(x[o:o+app.segment_size])
		app.plot()
		t_iteration_end = time.time()

		iteration_dur = t_iteration_end - t_iteration_st
		print(f"iteration took {iteration_dur*1000}ms ({iteration_dur/time_per_iteration*100:.2f}% of intended)")
		print("Pause", max(0.0, time_per_iteration - iteration_dur))
		plt.pause(max(0.0000001, time_per_iteration - iteration_dur))

		if app.closed:
			print("Window closed! Exit!")
			break
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


def calibrate(n_fft, overwrite=False):
	calibate_cache_path = Path("clips/app_calibration.json")
	data = cache_or(
		calibate_cache_path, 
		lambda: gender_scatter(n_fft, limit=40),
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

	# calibrate(1024)
	
	start_recording()
	# plot_file(Path("clips/z_trim.wav"))



if __name__ == "__main__":
	main()
