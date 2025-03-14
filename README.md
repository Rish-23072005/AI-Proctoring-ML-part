# Real-time Speech to Text (STT) using Faster WhisperModel and SpeechRecognition
This project provides a real-time speech-to-text (STT) solution using the Faster WhisperModel and the SpeechRecognition library. It captures audio from the microphone, transcribes it in real-time, and prints the transcriptions to the console.
## Features
- Real-time speech-to-text transcription
- Uses Faster WhisperModel for accurate transcription
- Handles ambient noise adjustment
- Supports multiple languages
- Configurable logging levels
## Requirements
- Python 3.10+
- pyaudio
- speechrecognition
- faster-whisper 
2. Ensure your CUDA drivers are up to date if you are using GPU for transcription. You can download the latest drivers from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).
## Usage
1. Run the [main.py](http://_vscodecontentref_/1) script:
    ```bash
    python main.py
    ```
2. The script will start listening to the microphone and transcribing speech in real-time. The transcriptions will be printed to the console.
3. To stop the transcription process, say "stop" or press `Ctrl+C`.
## Configuration
You can configure the STT object by modifying the parameters in the [__init__](http://_vscodecontentref_/2) method:
- [model_size](http://_vscodecontentref_/3): The size of the Whisper model to use (e.g., "medium.en").
- [device](http://_vscodecontentref_/4): The device to use for computation ("cuda" for GPU, "cpu" for CPU).
- [compute_type](http://_vscodecontentref_/5): The compute type to use (e.g., "float16").
- [language](http://_vscodecontentref_/6): The language for transcription (e.g., "en").
- [logging_level](http://_vscodecontentref_/7): The logging level for the application (e.g., "INFO").
## Troubleshooting
-ReadTimeoutError: Ensure your internet connection is stable. If the issue persists, download the model manually and load it from the local path.
- CUDA driver version is insufficient: Update your CUDA drivers to the latest version.
## Acknowledgements
- [Faster WhisperModel](https://github.com/openai/whisper)
- [SpeechRecognition](https://github.com/Uberi/speech_recognition)
