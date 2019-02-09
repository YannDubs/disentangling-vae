#! /bin/sh

mkdir -p data
cd data

python3 get_drive_file.py 0B7EVK8r0v71pZjFTYXZWM3FlRnM celebA.zip
mkdir celeba
unzip celebA.zip -d celeba/

cd ..

