# \# Face Recognition Project

# 

# \## Overview

# 

# This project implements a face recognition system leveraging deep learning technologies (MTCNN and FaceNet) and a K-Nearest Neighbors classifier. It allows users to upload images and recognize faces based on a pre-existing dataset.

# 

# \## Features

# 

# \* \*\*Face Detection\*\*: Utilizing MTCNN (Multi-task Cascaded Convolutional Networks).

# \* \*\*Embedding Generation\*\*: Face embeddings created using FaceNet (InceptionResnetV1).

# \* \*\*Classification\*\*: Face identification using K-Nearest Neighbors (KNN, k=1).

# \* \*\*User-Friendly Interface\*\*: Interactive image upload and immediate recognition results via Gradio.

# 

# \## Project Structure

# 

# ```

# Face Recognition Project

# │

# ├── dataset/                      # Dataset directory structured by person names

# │

# ├── embeddings.npy                # Saved embeddings (optional)

# │

# ├── labels.npy                    # Corresponding labels (optional)

# │

# ├── face\_recognition.ipynb        # Google Colab Notebook

# │

# └── README.md                     # Project Documentation

# ```

# 

# \## Installation and Setup

# 

# 1\. \*\*Clone the Repository\*\*:

# 

# ```bash

# git clone <repository-link>

# ```

# 

# 2\. \*\*Install Dependencies\*\*:

# 

# ```bash

# pip install facenet-pytorch torch torchvision sklearn opencv-python-headless matplotlib gradio

# ```

# 

# 3\. \*\*Dataset Preparation\*\*:

# &nbsp;  Organize images in the following format:

# 

# ```

# dataset/

# ├── Person1/

# │   ├── image1.jpg

# │   └── image2.jpg

# └── Person2/

# &nbsp;   ├── image1.jpg

# &nbsp;   └── image2.jpg

# ```

# 

# \## Running the Project

# 

# Open the provided Jupyter notebook `face\_recognition.ipynb` in Google Colab and execute each cell sequentially.

# 

# \### Interactive UI

# 

# The project includes a Gradio-based interface for uploading and recognizing faces:

# 

# ```python

# iface.launch(debug=True)

# ```

# 

# \## Evaluation

# 

# Evaluation metrics (accuracy, confusion matrix, classification report) are computed within the notebook to validate the performance of the recognition system.

# 

# \## Technologies Used

# 

# \* Python

# \* PyTorch

# \* Facenet-PyTorch

# \* Scikit-Learn

# \* Gradio

# \* Google Colab

