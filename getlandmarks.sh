#!/bin/bash
file=shape_predictor_68_face_landmarks.dat
if [ -f "$file" ]; then
echo "$file already exists"
else
wget https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat
fi