#! /bin/bash

mkdir -p model
cd model || exit 1
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
pip install -r requirements.txt
