# Local Setup Instructions for Skin Disease Detection App

## Prerequisites
- Python 3.8 or higher installed on your computer
- Internet connection for initial package installation

## Installation Steps

### 1. Download Project
- Download the ZIP file from Replit
- Extract it to a folder on your computer

### 2. Install Required Packages
Open Command Prompt (Windows) or Terminal (Mac/Linux) in the project folder and run:

```bash
pip install streamlit streamlit-webrtc torch torchvision opencv-python pillow numpy av
```

### 3. Run the Application
In the same terminal/command prompt, run:

```bash
streamlit run app.py
```

The app will open in your web browser at `http://localhost:8501`

## Usage
- Upload skin images or use your camera
- The AI model will analyze and provide predictions
- All processing happens locally on your device
- No internet required after initial setup

## Files Included
- `app.py` - Main application
- `model_utils.py` - AI model functions
- `image_processor.py` - Image processing
- `accuracy_optimizer.py` - Advanced prediction features
- `attached_assets/` - Contains your trained model and images
- `pyproject.toml` - Project dependencies

## Troubleshooting
- If packages fail to install, try: `pip install --upgrade pip` first
- For camera issues, ensure your browser allows camera access
- Model file should be in `attached_assets/` folder

## Privacy
- Everything runs locally on your device
- No data is sent to external servers
- Your images stay private on your computer