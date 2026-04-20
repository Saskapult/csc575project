# Real-time Spectral Voice Program
Real-time Spectral Voice Program (RSVP) is a tool to measure vocal resonance and vocal weight in real-time. 
This implementation is written using Python and runs locally on a user's machine. 
Future implementations aim to target the web, so stay tuned for that! 

## Setup 
- Install python 
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 
- Clone this repository 
- Run `sh librispeech.sh` to set up the LibriSpeech dataset 

## Usage 
Run `uv run app.py rt` to analyze your microphone input. 

Run `uv run app.py clip` to select a file to analyze. 
Alternatively, skip file selection by selecting the file in the program arguments, such as in `uv run app.py clip <clips/your_clip_name.wav>`. 
