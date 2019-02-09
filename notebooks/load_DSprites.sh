#! /bin/sh

# create data folder
mkdir -p 01_data
cd 01_data

git clone https://github.com/deepmind/dsprites-dataset.git
cd dsprites-dataset
rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5
cd ..
