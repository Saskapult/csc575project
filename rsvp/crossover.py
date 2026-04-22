from pathlib import Path
import numpy as np
import librosa
import matplotlib.pyplot as plt

from analysis import spectral_slope, vocal_resonance


def scatter_analysis(data_dir: Path, n_fft=2048):

	print("Analyze")
	# data = {i: {} for i in range(len(list(data_dir.iterdir())))}
	data = {}
	for file in data_dir.iterdir():
		index = int(file.name[0])
		example = file.stem.split("_")[1]
		print(example)

		x, srate = librosa.load(file)

		stft_freqs = librosa.fft_frequencies(sr=srate, n_fft=n_fft)
		stft = np.abs(librosa.stft(x, n_fft=n_fft))

		resonance, _ = vocal_resonance(stft, stft_freqs)
		slopes = -spectral_slope(stft, stft_freqs)
		
		if not (index in data.keys()):
			data[index] = {}
		data[index][example] = {
			"weight": slopes.tolist(),
			"resonance": resonance.tolist(),
		}
	
	print("Plot")
	fig, ax = plt.subplots()
	for i in range(len(data)):
		things = ["corrected", "target", "original"]
		# print(list(data[i].keys()))
		x = [np.mean(data[i][t]["weight"]) for t in things]
		y = [np.mean(data[i][t]["resonance"]) for t in things]

		# Original to target
		ax.annotate("", 
			xy=(x[1], y[1]), 
			xytext=(x[2], y[2]), 
			arrowprops=dict(arrowstyle="-", color='k', linestyle="--", alpha=0.5), 
			size=20,
		)
		# Original to corrected
		ax.annotate("", 
			xy=(x[0], y[0]), 
			xytext=(x[2], y[2]), 
			arrowprops=dict(arrowstyle="-", color='k', linestyle="-", alpha=0.5), 
			size=20,
		)

		ax.scatter(x, y)
		for t, xv, yv in zip(things, x, y):
			ax.annotate(t[0], (xv, yv))
		
	ax.set_xlabel("Weight")
	ax.set_ylabel("Resonance")
	
	plt.savefig("clips/crossover.png")
	# plt.show()


def main():
	scatter_analysis(Path("clips/r_data"))


if __name__ == "__main__":
	main()
