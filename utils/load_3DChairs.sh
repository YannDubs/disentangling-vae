#! /bin/sh

# create data folder
mkdir -p data
cd data
mkdir 3DChairs_64
cd 3DChairs_64
mkdir images_64 
cd ..

ls
# download 3D chairs
wget https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar
# extract the files
tar -xvf rendered_chairs.tar
root=rendered_chairs
new_root="3DChairs/images"
rm $root"/all_chair_names.mat"
mkdir -p $new_root
n=1
for dir in `ls -1t $root`; do
    for imgpath in `ls -1t $root/$dir/renders/*`; do
        imgname=$(echo "$imgpath" | cut -d"/" -f4)
        newpath=$img" "$new_root"/"$n"_"$imgname
        mv $imgpath $newpath
        n=$((n+1))
    done
done
rm -rf $root
cd ..
