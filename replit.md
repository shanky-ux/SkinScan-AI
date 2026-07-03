# Skin Disease Detection System

## Overview

A medical AI application built with Streamlit that detects and classifies skin diseases from images. The system uses a PyTorch-based deep learning model with ResNet architecture to analyze uploaded images or real-time camera feeds and provide skin condition predictions. The application emphasizes medical disclaimers and user safety by clearly stating it's for educational purposes only.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Single-page application with wide layout configuration
- **Real-time Camera Integration**: Uses streamlit-webrtc for live camera capture and processing
- **Interactive UI Components**: Image upload, camera capture, and results display with medical disclaimers
- **Session State Management**: Maintains captured images and prediction results across interactions

### Backend Architecture
- **PyTorch Deep Learning Model**: Custom SkinDiseaseModel class extending ResNet18 architecture
- **Image Processing Pipeline**: Comprehensive preprocessing with validation, resizing, normalization using ImageNet standards
- **Model Loading System**: Cached model loading with flexible checkpoint format handling
- **Modular Design**: Separated concerns across app.py (main UI), model_utils.py (ML operations), and image_processor.py (image handling)

### Data Processing
- **Image Preprocessing**: Standard medical image transforms including 224x224 resizing and ImageNet normalization
- **Input Validation**: Checks for minimum dimensions, valid color modes, and image integrity
- **Multi-format Support**: Handles RGB, RGBA, and grayscale images with automatic conversion

### Model Architecture
- **Base Model**: ResNet18 backbone modified for skin disease classification
- **Classification Head**: Custom fully connected layer for 7-class disease prediction
- **Transfer Learning**: Leverages pretrained weights adapted for medical imaging
- **CPU Inference**: Optimized for deployment without GPU requirements

## External Dependencies

### Core ML/AI Libraries
- **PyTorch**: Deep learning framework for model inference and neural network operations
- **torchvision**: Image transformations and pretrained model architectures
- **OpenCV (cv2)**: Computer vision operations and image processing utilities

### Web Framework
- **Streamlit**: Main web application framework with caching decorators
- **streamlit-webrtc**: Real-time camera streaming and WebRTC integration
- **aiortc (av)**: Audio/video processing for camera streams

### Image Processing
- **Pillow (PIL)**: Image manipulation and format conversion
- **NumPy**: Numerical operations for image arrays and tensor processing

### Model Assets
- **Pre-trained Model**: skin_disease_model_1755753853241.pth stored in attached_assets directory
- **Disease Classifications**: 7-class skin condition taxonomy (specific classes defined in model_utils)