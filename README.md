# Auditory Gender Analysis for Vocal Expression
<!-- Background -->
Many transgender people seek to alter their perceived vocal gender. 
The use of vocal therapy techniques is effective in altering perceived vocal gender. 
Financial and social barriers prevent many transgender people from accessing vocal therapy for this purpose. 
Many turn to online resources for gender classification, many of which solely measure vocal pitch. 
Focusing solely on altering vocal pitch results is not generally effective in altering perceived vocal gender. 
In contrast, vocal resonance and vocal weight are highly influential in the perceived gender of a speaker. 

<!-- Resonance and weight -->
Vocal resonance and vocal weight are alterable using voice therapy techniques. 
Learning to alter one's vocal resonance and vocal weight necessitates learning to identify these features in recorded audio. 
In the absence of an external judge, this is commonly performed by listening to one's own voice recordings. 
Many transgender people face difficulty in listening to voice recordings due to vocal dysphoria. 


## Katherine Smith V00907761 - Real-time Voice Space Program 

### Project Description 
<!-- Look upon my works -->
The Real-time Voice Space Program is a tool to measure vocal resonance and vocal weight in recorded speech. 
These data points are shown on a 2d graph, with vocal resonance as one axis and vocal weight as the other. 
The graphs depicted manually in [this video](https://www.youtube.com/watch?v=uVJuUoypVHE) may be considered a mock-up for the project, as well as some example audio for testing. 
These are displayed in real-time as a user speaks in order to better allow the user to learn to control specific muscle groups. 
Real-time operation is the key innovation of this project, as similar voice measurement tools [exist](https://acousticgender.space/) but do not operate in real-time. 

<!-- Advanced goals involve utilizing vocal resonance and vocal weight qualities as a means to identify similar voices. 
I hypothesize that this will select voices that are perceptually dissimilar aside from the qualities of vocal resonance and vocal weight, which may prove helpful in teaching a user to perceive these qualities.  -->

### Tools and Data Sets 
- Python for main codebase 
- Streamlit for UI  
- Rust for performance-critical code 
- [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and [LibriSpeech](https://www.openslr.org/12) for voice similarity 

### Timeline 
- Week 1 & 2: Set up environment, create framework for audio file input, create basic UI for audio clip recording. 
- Week 3 & 4: Extract vocal resonance measurement, display value as a UI element. 
- Week 5 & 6: Extract vocal weight measurement, display as UI element. 
- Week 7 & 8: Adapt previous algorithms to function using real-time audio input, display vocal resonance and weight in a 2d plot in real-time. 

### Objective 1: Offline Vocal Quality Measurement 
Create a web page that takes a voice clip as input and outputs vocal resonance and weight metrics. 

- PI1 (basic): Interface accepts recorded audio files clips. 
- PI2 (basic): Interface can record audio clips, which may then be used as input. 
- PI3 (expected): Interface displays vocal resonance metric. 
- PI4 (expected): Interface displays vocal weight metric. 
- PI5 (advanced): Vocal resonance and weight metrics are used to identify similar voices from a celebrity voice dataset to be used as examples. 

### Objective 2: Real-time Vocal Quality Measurement 
Objective 2 builds on objective 1 with a focus on real-time operation.  

- PI1 (basic): Interface records audio in real-time for immediate processing. 
- PI2 (basic): Measurement update rate is adjustable to maintain real-time feedback on less powerful hardware. 
- PI3 (expected): Vocal resonance and vocal weight metrics may be computed in real-time. 
- PI4 (expected): Vocal quality measurements are continuous over time (ie: do not jump from extremes during the course of a normally-spoken sentence). 
- PI5 (advanced): Real-time suggestion of similar voices based on vocal resonance and weight metrics. 


## Rayile (Raila) Adam V01073183 - Vocal Clip Correction 

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
* PI1 (basic): Build a web interface with simple "Record" and "Play" buttons.
* PI2 (basic): Display a visual comparison of Mel-spectrograms for the user and target clips.
* PI3 (expected): Create a button for pitch shifting that "fixes" the user's voice to match the target
* PI4 (expected): Provide a final "Mimicry Score" based on pitch accuracy and rhythmic alignment.
* PI5 (advanced): Implement high-quality correction to make the "corrected" voice sound natural.



# RESOURCES 



