# Deep Learning Feature Visualization with Grad-CAM

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

## Overview
This repository contains a complete PyTorch implementation of a Convolutional Neural Network (CNN) integrated with **Gradient-weighted Class Activation Mapping (Grad-CAM)**. 

While standard Deep Learning models act as "black boxes," providing predictions without explanations, this project implements a diagnostic visualization tool. By extracting the mathematical gradients from the final convolutional layer, the system generates a spatial heatmap (similar to a stress-concentration map in finite element analysis) that highlights the exact visual features the AI utilized to make its classification decision.

## Visual Results: AI Interpretability
*The images below demonstrate the model's prediction alongside the Grad-CAM heatmap, proving the model is focusing on the correct physical features of the object rather than background noise.*

![Grad-CAM Thermal Vision Diagnostics](assets/gradcam_output.jpg)

## Technical Implementation
This project is built using an Object-Oriented approach in Python, consisting of three main engineering pillars:

1. **Custom Data Pipeline (`torch.utils.data.Dataset`)**
   * Engineered a custom Dataset class to ingest raw images and labels.
   * Implemented dynamic data augmentation using `Albumentations` to prevent model overfitting and improve spatial generalization.
   * Managed tensor transformations and memory-efficient batch loading via PyTorch `DataLoader`.

2. **CNN Architecture & Training Optimization**
   * Built a custom multi-layer Convolutional Neural Network utilizing `torch.nn.Module`.
   * Designed a robust training loop calculating Cross-Entropy Loss.
   * Optimized network weights utilizing the Adam optimizer with a configured learning rate for steady gradient descent.

3. **Grad-CAM Extraction Logic**
   * Targeted the final spatial feature maps of the CNN architecture.
   * Computed the partial derivatives of the winning class score with respect to the feature map activations.
   * Applied a Rectified Linear Unit (ReLU) filter to isolate positive pixel contributions and upscaled the resulting matrix using OpenCV to overlay the diagnostic heatmap onto the original image.

## Libraries
* **Framework:** PyTorch (`torch`, `torch.nn`, `torchvision`)
* **Computer Vision:** OpenCV (`cv2`), Matplotlib (for spatial blending and rendering)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Pipeline:** Albumentations (Image augmentation)

## Acknowledgments
This repository was developed as part of the guided professional project *"Deep Learning with PyTorch: Grad-CAM"* provided by Coursera, featuring custom implementations of the dataset pipeline, training infrastructure, and mathematical gradient extraction functions.
