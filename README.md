# Galaxy Image Classifier

A FastAPI web application that classifies galaxy images using a deep learning model. The app provides both a web interface and a Gradio interface for galaxy image classification.

## Features

- Upload and classify galaxy images through a web interface
- Uses a pre-trained CNN model for galaxy classification
- Supports 37 different galaxy classification categories
- Real-time image processing and classification


## Installation

### 1. Download the Project

### 2. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Verify Model File

Make sure the model file exists in the correct location:
```
galaxy_classifier/
├── lab3_data/
│   └── galaxy_reduced_net.pth  # Pre-trained model file
```

## Running the Application

### Option 1: FastAPI Web Interface (Recommended)

1. **Start the server:**
   ```bash 
   uvicorn main:app --reload 
   ```

4. **Torchvision warnings:**
   - These are usually harmless warnings about image extensions
   - The application should still work normally

## Project Structure

```
galaxy_classifier/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── lab3_data/
│   └── galaxy_reduced_net.pth  # Pre-trained model
├── templates/
│   └── index.html         # Web interface template
└── static/                # Static files (CSS, JS, images)
```

## Model Information

The application uses a CNN (Convolutional Neural Network) model trained to classify galaxy images into 37 different categories including:
- Smooth and rounded galaxies
- Galaxies with features or disks
- Edge-on disk galaxies
- Spiral galaxies with various arm patterns
- Presence of odd features
