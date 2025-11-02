# Age‑Detection Model

A deep‑learning system for estimating a person’s age (and optionally gender) from facial images / video frames—built using pre‑trained convolutional neural networks, object detection (YOLO), and a voice‑based recognition component.

## 🚀 Table of Contents
- [About](#about)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Getting Started](#getting-started)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

## About  
This project aims to create a real‑time age (and optionally gender) detection system for facial images and live video streams. The primary goal is educational—developed as a final year engineering project using machine learning and computer vision.  
The system combines:  
- Face detection (via object detection, e.g., YOLOv5 or similar)  
- Age (and optionally gender) classification/regression using a pretrained CNN (e.g., ResNet50, EfficientNet)  
- An optional voice‐based module that allows age/gender recognition via audio input (if implemented)  
- A simple UI or demo interface to capture camera/video frames, process them, and display estimated age/gender.

## Features  
- Real‑time face detection in video/camera feed  
- Age estimation (in years or age‐group) from detected faces  
- Optional gender recognition (male/female)  
- Batch processing of images and single frame processing  
- Exportable result logs (e.g., CSV of image filename, detected age, optionally gender)  
- Easy to extend: plug in your own datasets, retrain model, fine‑tune for improved accuracy  
- Demo script to show live video inference (if included)  

## Tech Stack  
- **Language**: Python 3.x  
- **Libraries/Frameworks**:  
  - Computer Vision & Deep Learning: OpenCV, PyTorch/TensorFlow/Keras (depending on implementation)  
  - Pre‑trained CNN backbone: ResNet, EfficientNet, or similar  
  - Object detection: YOLO (or other face‑detection method)  
  - Audio processing for voice‑based component: (e.g., SpeechRecognition, librosa)  
- **Data**: Facial image datasets (e.g., IMDB‑WIKI, UTKFace) for age/gender training (if used)  
- **Hardware**: CPU or GPU (for faster inference/train)  
- **Dependencies**: Specified via `requirements.txt` (if present)

## Getting Started  
### Prerequisites  
- Python 3.x installed (recommend 3.8+).  
- A working camera/webcam (for live demo).  
- (Optional) GPU and CUDA setup for faster processing.  
- Install required Python packages:  
  ```bash
  pip install -r requirements.txt
  ```  
  If there is no `requirements.txt`, manually install the key libraries (e.g., OpenCV, PyTorch/Keras, etc.).  

### Installation & Setup  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/tharunrajendiran/Age‑detection‑model.git
   ```  
2. **Navigate into project directory**  
   ```bash
   cd Age‑detection‑model
   ```  
3. **Download/prepare model weights**  
   - If the pretrained model weights are included → verify path.  
   - If you need to train/finetune yourself → see Training section below.  
4. **Run the demo/inference script**  
   ```bash
   python demo.py
   ```  
   (Replace `demo.py` with the actual script name used.)  
5. **For training** (optional)  
   ```bash
   python train.py --dataset /path/to/faces --epochs 30 --batch_size 32
   ```  
   (Replace with your actual training script and parameters.)

## Usage  
- Launch `demo.py` (or equivalent) to open camera feed.  
- The system will detect faces, estimate age (and optionally gender), and overlay results on the video frames.  
- For image batch mode, run:  
  ```bash
  python infer_images.py --images /path/to/images
  ```  
- Modify configuration (e.g., `config.json`) to change parameters like detection threshold, face size, output format.  
- To use voice‑based recognition (if implemented):  
  ```bash
  python voice_recognition.py
  ```  
  Speak into the microphone, and the system will attempt to infer age/gender from voice.

## Results  
- Accuracy metrics: Report the performance (e.g., MAE for age estimation, accuracy for gender).  
- Sample output: Provide screenshots or videos of inference results.  
- Example:  
  ```text
  Image: face1.jpg → Predicted age: 28 years  
  Image: face2.jpg → Predicted gender: Male, age: 35 years
  ```  
- Known limitations:  
  - Works best with frontal faces under good lighting.  
  - Age estimation in large age‑ranges (>60 years) may be less accurate.  
  - Voice‑based component (if any) may perform poorly depending on microphone quality and environment.

## Project Structure  
```
Age‑detection‑model/
├── data/                    # Dataset folder (if included or for download instructions)  
├── models/                  # Pretrained models/weights  
├── src/                     # Main source code  
│   ├── face_detector.py     # Face detection logic  
│   ├── age_estimator.py     # Age estimation model code  
│   ├── gender_classifier.py # (optional) Gender classification code  
│   ├── voice_module.py      # (optional) Voice‑based recognition logic  
│   ├── demo.py              # Script for live camera demo  
│   ├── infer_images.py      # Script for batch image inference  
│   └── train.py             # Script to train/finetune model  
├── requirements.txt         # Python dependencies  
└── README.md                # This file  
```

## Contributing  
If you’d like to contribute improvements or new features (such as better UI, REST API interface, mobile deployment, better dataset augmentation), you’re very welcome!  
Please fork the repo, create a branch, make your changes, then submit a pull request.

## License  
Specify your license here (for example, MIT License):  
```text
MIT License  
Copyright (c) 2025 Tharun R  
```
