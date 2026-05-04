# AI Aimbot Assistand

## Introduction:
    The AI aimbot assistant is based on the YOLOv10 model, trained to obtain enemy positions through real-time high-speed screenshots. It then uses the Leonardo development version to simulate mouse input signals, allowing the crosshair to snap to the enemy, achieving an aimbot effect.

## Features

- **Real-Time Target Detection**: Uses YOLOv10 model to detect enemy positions through high-speed screenshots.
- **Precise Aiming**: Automatically adjusts the crosshair to lock onto enemy targets.
- **Mouse Input Simulation**: Simulates mouse movements to help players aim more accurately.
- **Optimized Performance**: Designed to work seamlessly with minimal delay during gameplay.

## Technologies Used

- **YOLOv10**: A deep learning model used for object detection, specifically for detecting enemy positions.
- **Leonardo Development Version**: Used for simulating mouse input signals.
- **Python**: Programming language used for development.
- **OpenCV**: For image processing and real-time screenshot capture.
- **NVIDIA Drivers**: For GPU acceleration and efficient computation.


## Usage Instructions

To use the AI Aimbot Assistant, please follow these steps to set up the necessary environment and run the project:

### 1. **Set Up NVIDIA Driver Environment**

Ensure that you have the correct **NVIDIA drivers** installed on your system to enable GPU acceleration. Follow these steps:

- Download and install the latest **NVIDIA drivers** for your GPU from the [NVIDIA official website](https://www.nvidia.com/Download/index.aspx).
- If you are using CUDA or cuDNN for deep learning, make sure you have those installed as well. You can follow the installation guide on the official [CUDA Toolkit page](https://developer.nvidia.com/cuda-toolkit) and [cuDNN page](https://developer.nvidia.com/cudnn).

### 2. **Install Project Dependencies**

After setting up your environment, you need to install the required libraries and dependencies for the project. Run the following command to install them:

```bash
pip install -r requirements.txt
```

### 3. **Embed PIDc++ into the Development Board**
The PIDc++ library is commonly used for embedded control systems, such as adjusting the movement of robots or other systems that require control algorithms. Through Arduino IDE, you can upload this library onto the development board to enable control.
