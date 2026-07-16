# AI-Powered Apparel Sorting System

## Project Overview
This project is an automated apparel classification system developed for e-commerce warehouse logistics. Utilizing a custom Convolutional Neural Network (CNN) architecture, this system categorizes high-definition (128x128 pixel) RGB clothing images into 5 distinct categories: **Jacket, Polo shirt, Shirt, T-shirt, and Tank top.**

The system is built with a modular "Separation of Concerns" approach, ensuring scalability and ease of debugging. It features a custom data pipeline for dynamic Kaggle dataset ingestion, a modular training architecture, and an interactive Gradio web interface for real-time inference.

## Prerequisites
Before you begin, ensure you have the following installed on your machine (macOS or Windows):

1. **Python 3.11+**
2. **Conda** (Miniconda or Anaconda recommended)
3. **Kaggle Account** (To download the dataset)

## Installation Instructions

### 1. Set Up the Environment
Open your terminal (or Command Prompt on Windows) and navigate to the project folder. Create and activate your virtual environment:

```bash
## Create the environment
conda create -n csc583_env python=3.11 -y

## Activate the environment
conda activate csc583_env

### 2. Install Dependencies
pip install tensorflow gradio numpy matplotlib seaborn pillow kaggle

### 3. Configure Kaggle API
1. Log into your Kaggle account.
2. Go to Account Settings -> API -> Create New Token.
3. Copy the Kaggle key to extract_images.py

### Running the Application
Step 1: Download & Organize Dataset
bash -> python extract_images.py

Step 2: Launch the GUI
bash -> python main.py

Note: A local URL (usually http://127.0.0.1:7860) will appear in the terminal. Open this URL in your web browser to start sorting your apparel items.

Project Structure
extract_images.py: Handles Kaggle API authentication, dataset downloading, and automated folder partitioning.

data_handler.py: Parses the local directory, resizes images to 128x128 RGB, and normalizes pixel data.

model_builder.py: Defines the CNN layer architecture (Conv2D, Pooling, Dense, Softmax).

trainer.py: Executes the model training, model evaluation, and generates the Confusion Matrix.

main.py: The entry point. Manages the Gradio interface and inference logic.

Troubleshooting
ModuleNotFoundError: If you encounter this error even after installation, ensure your terminal is active in (csc583_env) and run: python -m pip install [library_name].

ZSH/Semaphore Warnings: If you see "leaked semaphore" messages when closing the program, this is a normal byproduct of forcefully terminating the Python process and can be safely ignored.