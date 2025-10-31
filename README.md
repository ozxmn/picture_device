# Picture device detector

This project is a desktop application that identifies the device used to capture a digital image by analyzing image characteristics using multiple methods.

## Overview

The application performs three independent analyses to estimate the most probable source device of a photo:

1. **JPEG Compression analysis** – examines quantization matrices and DCT artifacts.
2. **PRNU (Photo response non-uniformity)** – extracts and matches sensor noise patterns unique to each device.
3. **CNN-Based noise residual analysis** – uses a trained convolutional neural network to classify noise features.

The results from these three methods are fused to produce a combined prediction of the device using which the picture was taken.

## Features

* Train models from image folders organized by device type.
* Perform device detection on images with metadata removed.
* View detailed results for each method and combined output.
* Multithreaded training and analysis with progress and log display.

## Project Structure

```
picture_device/
├── cnn_model.py
├── database.py
├── forensic_analysis.py
├── fusion_analysis.py
├── gui.py
├── jpeg_analyzer.py
├── main.py
├── prnu_analyzer.py
├── utils.py
├── requirements.txt
└── models/ (generated after the training)
```

## Usage

1. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the application:

   ```
   python main.py
   ```

3. **Training**

   * Select a folder with images.
   * Press "Start training" button to generate reference models for each analysis method.

4. **Detection**

   * Select an image for testing.
   * Press "Analyze image" button to see the results.

## Authors

Mansur 

[GitHub link](https://github.com/ozxmn)

