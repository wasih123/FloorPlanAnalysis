#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e 

echo "Installing Kaggle CLI..."
pip install kaggle

echo "Creating data directory..."
mkdir -p ../data/cubicasa5k
cd ../data/cubicasa5k

echo "Downloading CubiCasa5k dataset from Kaggle..."
# This uses the Kaggle API to download the specific dataset
kaggle datasets download -d qmarva/cubicasa5k

echo "Unzipping dataset (this may take a while)..."
# -q makes it quiet so it doesn't flood your terminal
unzip -q cubicasa5k.zip 

echo "Cleaning up the zip file to save server space..."
rm cubicasa5k.zip

echo "Download and extraction complete! Your data is ready."