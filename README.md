# csc575project

Here are some ideas and thoughts that I have had. 
Feel free to edit or add to or this!

Also I *love* funky names. 
I spent way too long thinking of Auditory Gender Analysis for Vocal Expression (AGAVE). 
Please let me knwo if you also have funky name ideas. 
It will be fun! 

## Deliverable Idea 1 - Real-time Voice Space Program 
This was my initial idea for the project. 
It is to find measures for various qualities of voice such as vocal resonance and vocal weight and then put them on a graph for visualization. 
Something like this already [exists](https://acousticgender.space/) but this one could operate in real-time and include additional vocal metrics. 
[This video](https://www.youtube.com/watch?v=uVJuUoypVHE) shows some of the vocal qualities to measure (and it’d be really cool to use the audio from it for testing!). 

I was thinking that I’d implement this as a web page. The computation could be performed with rust compiled to WASM but that’s mostly because I want to write some rust code. Otherwise javascript would probably work too. 

One key part of this is that I want it to run in real-time. 
I find that helpful and I can probably find some papers to back it up. 

I (Katherine) want to work on this but I am open to working on other things. 

## Rayile Adam Deliverable Idea 2 - Vocal Clip Correction 
(It records one’s voice and then alters it and then plays the altered version back. Maybe with a configurable delay? The idea is that a person can try to mimic what they hear. )

### Project Description
This project focuses on building a "Mimicry Trainer" that records a user's voice and compares it to a target vocal clip. It analyzes pitch ($f_0$) and rhythm to provide feedback and uses signal processing to "correct" the user's voice to match the target's characteristics.

### Tools and Data Sets
* Framework: PyTorch & TorchAudio for handling the audio math, sound processing and feature extraction.
* Analysis: Librosa for implementation of the Dynamic Time Warping (DTW) algorithm to help align the timing of the two voices.
* UI/Deployment: Streamlit for a web-based interactive dashboard.
* Data Sets: LibriSpeech. For high-quality reference speech samples.
* CREMA-D: To provide varied emotional and tonal reference clips for testing mimicry across different vocal weights and resonances.


### Timeline
* Week 1 & 2: Set up PyTorch/TorchAudio environment; build basic recording/playback in Streamlit.
* Week 3 & 4: Implement pitch detection and visualize Mel-spectrograms.
* Week 5 & 6: Implement DTW to align clips and calculate the similarity score (MSE).
* Week 7 & 8: Build the "Correction" module for pitch shifting and finalize the report.

### Objective 1: Implement an Automated Pitch Comparison Pipeline
* PI1 (basic): Successfully load audio files into PyTorch tensors using TorchAudio.
* PI2 (basic): Extract pitch frequency from recordings to visualize the "notes".
* PI3 (expected): Use DTW to line up the user's voice with the target voice in time.
* PI4 (expected): Calculate an accuracy score MSE showing the difference between voices.
* PI5 (advanced): Integrate a pre-trained model (e.g., Wav2Vec2) to improve alignment for better scoring.

### Objective 2: Develop a Vocal Correction and Feedback System
This objective focuses on making the app easy to use for everyone.
* PI1 (basic): Build a web interface with simple "Record" and "Play" buttons.
* PI2 (basic): Display a visual comparison of Mel-spectrograms for the user and target clips.
* PI3 (expected): Create a button for pitch shifting that "fixes" the user's voice to match the target
* PI4 (expected): Provide a final "Mimicry Score" based on pitch accuracy and rhythmic alignment.
* PI5 (advanced): Implement high-quality correction to make the "corrected" voice sound natural.


## Deliverable Idea 3 - Finding Similar Voices (with Machine Learning?) 
This could aim to compute voice similarity. One could arrange voices by some similarity metric. 

## Deliverable Idea 4 - An Empty One 
Feel free to add to these! 

# Information that could be useful maybe 
[VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
- Dataset of celebrity voices 

Jess Gibbard
- Does voice lessons, we could try contacting her for info (I’ve already asked her about measuring things using a spectrogram and she had useful info)

[Real-Time Resonance Biofeedback for Gender-Affirming Voice Training: Usability Testing of the TruVox Web-Based Application](https://pubmed.ncbi.nlm.nih.gov/41176464/) 
- Measures efficacy of a real-time training tool. Similar to idea 1 in concept

[Web-Based Application for Real-Time Biofeedback of Vocal Resonance in Gender-Affirming Voice Training: Design and Usability Evaluation](https://www.isca-archive.org/interspeech_2025/mcallister25_interspeech.pdf) 
- Measures the efficacy of an online resonance training tool. Like idea 1

[Gender-Affirming Voice Training for Trans Women: Effectiveness of Training on Patient-Reported Outcomes and Listener Perceptions of Voice](https://pubs.asha.org/doi/10.1044/2023_JSLHR-23-00258) 
- Shows the efficacy of voice training techniques 


