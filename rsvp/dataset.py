

# From libri into data pls 
# Make into wav too 

from pathlib import Path
import ffmpeg


dir_f = Path("./data/f")
dir_m = Path("./data/m")

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

		# # Get all files 
		# for i, f in enumerate(id.glob("**/*.flac")):
		# 	out_name = base/f"{s}_{i}.wav"
		# 	if not out_name.exists():
		# 		ffmpeg.input(f).output(str(out_name)).run()



if __name__ == "__main__":
	extract_librispeech_data(Path("./LibriSpeech"))
