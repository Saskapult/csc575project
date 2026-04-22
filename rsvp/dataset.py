from pathlib import Path
import shutil
import sys
import ffmpeg
import pandas as pd
from tqdm import tqdm

dir_base = Path("./data2")
dir_f = dir_base/"f"
dir_m = dir_base/"m"

dir_base.mkdir(exist_ok=True)
dir_f.mkdir(exist_ok=True)
dir_m.mkdir(exist_ok=True)


def extract_librispeech_data(path: Path):
	with open(path/"SPEAKERS.TXT") as fp:
		lines = fp.readlines()
	lines = [l for l in lines if not l.startswith(";")]
	lines = [l for l in lines if l.strip() != ""]

	tuples = [l.split("|") for l in lines]
	tuples = [[t.strip() for t in tup] for tup in tuples]
	
	gender_by_id = {t[0]: (t[1], t[4]) for t in tuples}

	i_f = 0
	i_m = 0
	for id in (path/"dev-clean").iterdir():
		print(id, gender_by_id[id.stem])

		g, s = gender_by_id[id.stem]

		base = dir_f if g == "F" else dir_m
		if g == "F":
			i_f += 1
		else:
			i_m += 1
		print(i_f, i_m)

		# Get all files 
		for i, f in enumerate(id.glob("**/*.flac")):
			out_name = base/f"{s}_{i}.wav"
			if not out_name.exists():
				ffmpeg.input(f).output(str(out_name)).run()


def extract_cremad_data(path: Path):
	actors = pd.read_csv(path/"VideoDemographics.csv")
	actors = actors.set_index("ActorID")
	print(actors)
	print(actors.dtypes)

	for file in tqdm((path/"AudioWAV").iterdir()):
		actor_id = int(file.name[:4])
		gender = actors.loc[actor_id]
		gender = gender["Sex"][0].lower()

		i = 0
		while True:
			out_file = dir_base/gender/f"{actor_id}_{i}.wav"
			if not out_file.exists():
				break
			else:
				i += 1
		
		print(f"{file} -> {out_file}")
		shutil.copy(file, out_file)


def main():
	if sys.argv[1] == "librispeech":
		extract_librispeech_data(Path("./clips/LibriSpeech"))
	elif sys.argv[1] == "cremad":
		extract_cremad_data(Path("./clips/crema-d-mirror"))
	else: 
		print("Input a valid dataset name!")
		exit(1)


if __name__ == "__main__":
	main()
