# AI-Based Oil Spill Detection and Cleanup System

## Overview

The AI-Based Oil Spill Detection and Cleanup System is an autonomous solution designed to detect and assist in the cleanup of oil spills using Artificial Intelligence, Computer Vision, Embedded Systems, and IoT technologies.

The system uses a YOLOv11m-seg instance segmentation model for oil spill detection and segmentation. The trained model is used for PC-based inference, while a Raspberry Pi is used for boat control and hardware interfacing. The system also incorporates an oil spill cleanup mechanism to support automated spill collection.

## Objectives

- Detect oil spills using computer vision and deep learning.
- Segment oil-contaminated regions using YOLOv11m-seg.
- Integrate AI-based detection with an autonomous boat platform.
- Control the boat using Raspberry Pi-based embedded systems.
- Integrate an oil spill cleanup mechanism.
- Provide a foundation for IoT-enabled monitoring and environmental surveillance.

## Key Features

- Oil spill detection and segmentation using YOLOv11m-seg
- Computer vision-based image processing
- PC-based model inference
- Raspberry Pi-based boat control
- Integration of motors, sensors and control electronics
- Autonomous oil spill cleanup mechanism
- Roboflow-based annotated dataset management
- Google Colab-based model training
- Model evaluation using detection results and confusion matrix
- Supporting CAD, circuit, project and research documentation

## System Architecture

The system consists of the following major components:

- Camera Module
- PC-based AI Inference
- YOLOv11m-seg Model
- Raspberry Pi
- Motor Drivers
- Propulsion Motors
- Oil Spill Cleanup Mechanism
- Battery and Power System
- IoT-based Monitoring and Communication

## Repository Structure

```text
AI-Based-Oil-Spill-Detection-and-Cleanup/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── Deployment/
│   ├── Pi_code.py
│   └── control-boat.py
│
├── Docs/
│   ├── Block_Diagram.png
│   ├── CAD_labelled_view.jpg
│   ├── CAD_final_design_page-0001.jpg
│   ├── Circuit_Design.jpg
│   ├── One page abstract.docx
│   ├── Project PPT.pptx
│   ├── Project report.docx
│   └── Project_Conference_Paper.docx
│
├── Model Training/
│   ├── Model_Train.ipynb
│   └── Model_Train.py
│
├── Results/
│   ├── Image 1.jpeg
│   ├── Image 2.jpeg
│   ├── Image 3.jpeg
│   ├── Image 4.jpeg
│   ├── Image 5.jpeg
│   ├── Image 6.jpeg
│   └── confusion_matrix_yolov11m.png
│
└── Unlabelled Sample Dataset/
    ├── image_01.jpg
    ├── image_02.jpg
    ├── image_03.jpg
    ├── image_04.jpg
    ├── image_05.jpg
    ├── image_06.jpg
    └── image_07.jpg
```

## Deployment

The `Deployment` directory contains the code used for the operational system.

### Raspberry Pi Code

`Pi_code.py` contains the code intended to run on the Raspberry Pi.

The Raspberry Pi is responsible for boat control and interaction with the connected hardware components.

### PC-Based Inference

`control-boat.py` contains the PC-side inference implementation.

The PC loads the trained YOLOv11m-seg model and performs oil spill detection and segmentation on input images.

The PC-based inference and Raspberry Pi control code are maintained separately to keep the AI inference and embedded hardware control components modular.

## Model Training

The `Model Training` directory contains the Google Colab Jupyter Notebook and Python training script used to train the model.

```text
Model Training/
├── Model_Train.ipynb
└── Model_Train.py
```

The training workflow uses the Roboflow API to access the annotated dataset used for model training.

### Training Workflow

1. Connect to the Roboflow project using the Roboflow API.
2. Access the annotated oil spill dataset.
3. Download the dataset in YOLO format.
4. Load the YOLOv11m-seg pretrained model.
5. Configure the training parameters.
6. Train the model using Google Colab GPU acceleration.
7. Validate the trained model.
8. Generate training and validation metrics.
9. Generate model evaluation results.
10. Save the trained model and training outputs.

### Training Configuration

- Model: YOLOv11m-seg
- Task: Instance Segmentation
- Image Size: 640 × 640
- Epochs: 100
- Batch Size: 16
- Optimizer: AdamW
- Early Stopping Patience: 25
- Training Environment: Google Colab
- Dataset Platform: Roboflow

The training notebook also contains the training curves and evaluation outputs generated during the training process.

## Dataset

The annotated dataset used for training is hosted and managed on Roboflow.

The complete annotated dataset is not stored in this GitHub repository. Instead, the training notebook uses the Roboflow API to access the annotated dataset.

### Dataset Link

[Roboflow Dataset](https://universe.roboflow.com/oil-spill-detection-mlacj/oil-xzcql-jz6dq)

### Dataset Format

YOLO

### Class

- Oil Spill

## Unlabelled Sample Dataset

The `Unlabelled Sample Dataset` directory contains seven unannotated images provided as reference/sample input data.

These images are included to demonstrate the type of input that can be provided to the detection system.

```text
Unlabelled Sample Dataset/
├── image_01.jpg
├── image_02.jpg
├── image_03.jpg
├── image_04.jpg
├── image_05.jpg
├── image_06.jpg
└── image_07.jpg
```

These seven images are **not the annotated training dataset**.

The annotated training data is hosted on Roboflow and can be accessed through the Roboflow API during model training.

## Results

The `Results` directory contains sample outputs generated by the trained YOLOv11m-seg model along with model evaluation results.

It includes:

- Oil spill detection outputs
- Segmentation results
- Sample model predictions
- Confusion matrix

### Detection Results

The included output images demonstrate the model's ability to identify and segment oil-contaminated regions from input images.

### Confusion Matrix

The file:

```text
Results/confusion_matrix_yolov11m.png
```

contains the confusion matrix generated during model evaluation.

The results provide a visual and quantitative reference for evaluating the trained model.

## Documentation

The `Docs` directory contains the supporting project documentation and design materials.

It includes:

- System block diagram
- CAD labelled view
- Final CAD design
- Circuit design
- One-page project abstract
- Project presentation
- Project report
- Conference paper

These documents provide additional information about the system architecture, hardware design, implementation and research work carried out as part of the project.

## Technology Stack

### Programming

- Python

### Artificial Intelligence and Computer Vision

- YOLOv11m-seg
- Ultralytics
- PyTorch
- OpenCV
- NumPy

### Embedded Systems

- Raspberry Pi
- Motor Drivers
- Propulsion Motors
- Sensors
- Camera Module
- Battery and Power System
- Cleanup Mechanism

### IoT

- IoT-based monitoring and communication

### Development and Dataset Tools

- Google Colab
- Roboflow
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/Varunesh-S/AI-Based-Oil-Spill-Detection-and-Cleanup.git
```

Navigate to the project directory:

```bash
cd AI-Based-Oil-Spill-Detection-and-Cleanup
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### PC-Based Inference

The PC-based inference code is located in:

```text
Deployment/control-boat.py
```

Run the inference script using:

```bash
python Deployment/control-boat.py
```

The script loads the trained YOLOv11m-seg model and performs oil spill detection on the provided input.

### Raspberry Pi Deployment

The Raspberry Pi control code is located in:

```text
Deployment/Pi_code.py
```

This code is intended to run on the Raspberry Pi and is responsible for controlling the boat and interfacing with the connected hardware components.

## Model Training

To train the model using the provided training workflow:

1. Open `Model Training/Model_Train.ipynb`.
2. Open the notebook in Google Colab.
3. Enable GPU acceleration.
4. Configure the Roboflow API credentials securely.
5. Access the annotated dataset through Roboflow.
6. Execute the training workflow.
7. Validate the trained model.
8. Review the generated training curves and evaluation results.

### Roboflow API Security

The Roboflow API key should not be committed to the GitHub repository.

Use your own API key when running the training notebook and keep credentials private.

## Results Summary

The trained model demonstrates the ability to detect and segment oil spill regions from input images.

The repository contains:

- Sample detection outputs
- Segmentation results
- Confusion matrix
- Training curves in the model training notebook

The current project achieved an approximate detection accuracy of **74%** based on the project evaluation.

## Applications

The system can be applied to:

- Marine pollution monitoring
- Oil spill detection
- Environmental surveillance
- Smart water monitoring
- Autonomous surface vehicles
- Oil spill response and management
- AI-assisted environmental cleanup

## Future Enhancements

Potential future improvements include:

- Improve detection accuracy using larger and more diverse datasets.
- Deploy the optimized model directly on edge hardware.
- Implement real-time camera-based detection.
- Integrate GPS-based autonomous navigation.
- Implement real-time oil spill location tracking.
- Develop a cloud-based monitoring dashboard.
- Improve autonomous navigation and obstacle avoidance.
- Improve the oil collection mechanism.
- Integrate additional environmental sensors.
- Extend the system to support multiple autonomous cleanup units.

## Academic Project

This project was developed as an academic project combining:

- Artificial Intelligence
- Computer Vision
- Embedded Systems
- IoT
- Autonomous Systems
- Hardware Control
- Environmental Monitoring

The project demonstrates the integration of an AI-based oil spill detection system with an autonomous boat platform to support oil spill monitoring and cleanup.

## Author

**Varunesh S**

Email: [varunesh1000@gmail.com](mailto:varunesh1000@gmail.com)

LinkedIn: [https://www.linkedin.com/in/varunesh-s/](https://www.linkedin.com/in/varunesh-s/)

GitHub: [https://github.com/Varunesh-S](https://github.com/Varunesh-S)

## License

This project is licensed under the MIT License.
