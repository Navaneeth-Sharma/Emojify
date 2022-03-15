#!/bin/bash

mkdir -p ~/.kaggle || echo "The ~./kaggle directory is present"
mv kaggle.json ~/.kaggle/kaggle.json || echo "The kaggle.json is moved"
ls ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

# $DATASET="msambare/fer2013"

if [ -d Data ]; then
    python data_preparation.py
    sleep 10
else
    kaggle datasets download -d msambare/fer2013

    mkdir Data
    mv fer2013.zip Data/fer2013.zip
    unzip Data/fer2013.zip -d Data/
fi
