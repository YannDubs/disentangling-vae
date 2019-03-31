import PIL
from PIL import Image
import os
from math import ceil, floor

IMGS_DIR = "imgs"


def concatenate_images(image_list, width, height, nr_columns = 1, percentage_space = 0.0):
    total_nr_images = len(image_list)  
    nr_rows = int(ceil(float(total_nr_images)/nr_columns))
    space = int(round(percentage_space*max(width, height)))
    new_image = PIL.Image.new("RGB", (width*nr_columns, height*nr_rows))
    # === create new image === # 
    for id_image in range(len(image_list)):
        image = image_list[id_image]
        id_x = id_image%nr_columns
        id_y = floor(id_image/nr_columns)
        new_image.paste(image, ((width+space)*id_x, (height+space)*id_y))
    return new_image

def get_image_list(image_file_name_list):
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(os.path.join(IMGS_DIR, file_name)))
    return image_list

def get_max_size(image_list):
    width_max = 0;
    height_max = 0;
    for image in image_list:
        image_size = image.size
        width_max = max(image_size[0],width_max)
        height_max = max(image_size[1],height_max)
    return width_max, height_max

def main():
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    image_file_name_list = [
    'show_disentanglement.png',
    'show_disentanglement.png',
    'show_disentanglement.png'
    ]
    output_file_name = 'combined_image'
    
    nr_columns = 2

    image_list = get_image_list(image_file_name_list)
    width_max, height_max = get_max_size(image_list)
    percentage_space  = 0.03
    new_image = concatenate_images(image_list, width_max, height_max, nr_columns, percentage_space)
    
    new_image.save(os.path.join(IMGS_DIR, output_file_name + '.png'))
    return True

if __name__ == '__main__':
    main()
