# AI-Based Oil Spill Detection and Cleanup System

## Overview

The AI-Based Oil Spill Detection and Cleanup System is an autonomous solution designed to detect and assist in the cleanup of oil spills using Artificial Intelligence, Computer Vision, Embedded Systems, and IoT technologies. The system leverages the YOLOv11 object detection model for real-time oil spill identification and integrates Raspberry Pi-based edge computing with an IoT-enabled monitoring platform to support remote operation and environmental surveillance.

## Objectives

- Detect oil spills in real time using computer vision.
- Enable autonomous cleanup through an integrated cleanup mechanism.
- Provide remote monitoring using IoT technologies.
- Develop a scalable edge-cloud architecture for environmental monitoring.

## Key Features

- Real-time oil spill detection using YOLOv11
- Computer vision-based image processing with OpenCV
- Raspberry Pi-based edge deployment
- IoT-enabled remote monitoring and control
- Autonomous cleanup mechanism integration
- Image and video inference support

## System Architecture

The system consists of the following major components:

- Camera Module
- Raspberry Pi
- YOLOv11 Detection Model
- OpenCV Image Processing
- Cleanup Mechanism
- Remote Monitoring Dashboard

## Technology Stack

### Programming Language
- Python

### Artificial Intelligence and Computer Vision
- YOLOv11
- OpenCV
- PyTorch

### Embedded Systems
- Raspberry Pi
- Propulsion Motors
- Skimmer Motor
- BMS
- Battery Source
- Pi Camera

### Internet of Things
- IoT-based Remote Monitoring

### Development Tools
- Google Colab
- Roboflow
- CUDA (TensorRT)
  
## Dataset

The dataset used for this project is hosted on Roboflow and consists of annotated oil spill images prepared for training YOLO-based computer vision models.

The project was trained using a custom dataset managed on Roboflow.

A subset of sample images is included for demonstration purposes.

Dataset Link:
https://universe.roboflow.com/oil-spill-detection-mlacj/oil-xzcql-jz6dq

Dataset Format:
YOLO

Class:
- Oil Spill

## Model Training

The YOLOv11 model was trained using Google Colab with GPU acceleration.

Training Framework:
- Ultralytics YOLOv11

Environment:
- Google Colab
- Python
- PyTorch

Dataset Source:
- Roboflow

Model:
- YOLOv11

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/AI-Oil-Spill-Detection-System.git
```

Navigate to the project directory:

```bash
cd AI-Oil-Spill-Detection-System
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the detection script:

```bash
python detect.py
```

## Results

- Real-time oil spill detection using YOLOv11
- Detection accuracy of approximately 74%
- Integrated IoT-based remote monitoring
- Autonomous cleanup mechanism for spill collection

## Applications

- Marine pollution monitoring
- Environmental protection
- Smart water surveillance
- Autonomous surface vehicles
- Oil spill response and management

## Future Enhancements

- Improve detection accuracy using larger and more diverse datasets
- Integrate GPS-based autonomous navigation
- Optimize inference for edge deployment
- Develop a cloud-based monitoring dashboard
- Extend support for multiple autonomous cleanup units

## Author

**Varunesh S**

B.Tech – Electrical and Electronics Engineering

Email: varunesh1000@gmail.com

LinkedIn: https://www.linkedin.com/in/varunesh-s/

GitHub: https://github.com/Varunesh-S

## License

This project is licensed under the MIT License.
