# Real-time Spectral Voice Program
Real-time Spectral Voice Program (RSVP) is a tool to measure vocal resonance and vocal weight in real-time. 
This implementation is written using Python and runs locally on a user's machine. 
Future implementations aim to target the web, so stay tuned for that! 

## Setup 
- Install python 
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 
- Clone this repository 
- Run `sh cremad.sh` to set up the CREMA-D dataset (note: this will take quite a long time!)
	- Alternatively, run `sh librispeech.sh` to set up the LibriSpeech dataset 

## Usage 
Run `uv run app.py rt` to analyze your microphone input. 

Run `uv run app.py clip` to select a file to analyze. 
Alternatively, skip file selection by selecting the file in the program arguments, such as in `uv run app.py clip <clips/your_clip_name.wav>`. 

Most systems should be able to process RSVP in real-time by default, but if computational power is limited the program may be made faster by increasing the `RSVP_N_FFT` (default 2048) environment variable. 
Alternatively, one can alter the `RSVP_UPDATE_RATE` (default 8) variable. 
