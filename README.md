# AI Proctoring System Real-time Facial Detection with Real-time Speech-to-Text (STT).

## Overview
This comprehensive AI Proctoring System integrates multiple advanced computer vision models along with real-time Speech-to-Text (STT) capabilities to monitor users during online examinations or restricted environments. The system combines:

✅ **Face Recognition** using DeepFace  
✅ **Body Posture Detection** with MediaPipe  
✅ **Gaze Tracking** for focus monitoring  
✅ **Object Detection** (YOLOv5 - Faster R-CNN) for identifying suspicious items like phones or laptops  
✅ **Real-time Speech-to-Text (STT)** using Faster WhisperModel and SpeechRecognition  

The system triggers alerts when anomalies are detected, ensuring enhanced security during examination processes.

---

## Features
- **Face Recognition**: Verifies if the detected face matches the authorized user database.
- **Body Posture Analysis**: Monitors the user's posture and alerts if unusual behavior is detected.
- **Gaze Tracking**: Tracks eye movements to identify prolonged distractions.
- **Object Detection**: Detects unauthorized objects such as phones, tablets, or laptops in the frame.
- Real-time Speech-to-Text (STT): Captures audio from the microphone and transcribes speech in real-time.
- Alert System: Triggers alerts for suspicious activities and keeps a count for monitoring purposes.

## Requirements
Before running the code, ensure you have the following dependencies installed:


pip install opencv-python mediapipe deepface gaze-tracking torch torchvision pyaudio speechrecognition faster-whisper

Additionally, download the YOLOv5 model weights:
- The `fasterrcnn_resnet50_fpn` model is included with `torchvision`, but ensure you have updated weights.

For GPU support:
- Ensure your CUDA drivers are up to date if you are using GPU for transcription. You can download the latest drivers from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).

## Configuration
You can configure the STT object by modifying the parameters in the `__init__` method:
- **`model_size`**: The size of the Whisper model to use (e.g., "medium.en").
- **`device`**: The device to use for computation ("cuda" for GPU, "cpu" for CPU).
- **`compute_type`**: The compute type to use (e.g., "float16").
- **`language`**: The language for transcription (e.g., "en").
- **`logging_level`**: The logging level for the application (e.g., "INFO").


## Usage Instructions
1. Ensure your webcam is properly connected.
2. Run the program. The system will:
   - Continuously analyze video feed.
   - Trigger alerts for any suspicious activity.
   - Transcribe speech in real-time and print transcriptions to the console.
3. Press **'Q'** to exit the system.

To stop the transcription process, say "stop" or press `Ctrl+C`.
## Troubleshooting
- **ReadTimeoutError**: Ensure your internet connection is stable. If the issue persists, download the model manually and load it from the local path.
- **CUDA driver version is insufficient**: Update your CUDA drivers to the latest version.
## Known Issues
- The object detection model may require fine-tuning for improved accuracy in various lighting conditions.
- The DeepFace model may misidentify faces if the image resolution is too low.
## Future Improvements
- Adding multi-face detection for identifying multiple participants simultaneously.
- Implementing an enhanced alert mechanism with real-time notifications.
- Integrating cloud storage for recording detected events.

## Acknowledgements
- [Faster WhisperModel](https://github.com/openai/whisper)
- [SpeechRecognition](https://github.com/Uberi/speech_recognition)


## Contributing
Contributions are welcome! Feel free to submit issues, feature requests, or pull requests to improve the system.

## Contact
For any questions or support, please reach out to this email[rishabhtripathi.9984@gmail.com].

