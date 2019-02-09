#! /bin/sh

mkdir -p 01_data
cd 01_data

python3 get_drive_file.py 0B7EVK8r0v71pZjFTYXZWM3FlRnM celebA.zip
mkdir celeba
unzip celebA.zip -d celeba/

cd ..

