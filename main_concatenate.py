import PIL
from PIL import Image
import os
from math import ceil, floor

IMGS_DIR = os.path.join('results', 'imgs_prior','Big_prior_traversals')


def concatenate_images(image_list, width, height, nr_columns = 1, percentage_space = 0.0):
    """ Concatenate the images in the image_list such that there are nr_columns amount of columns. The images can have different size. 
        The width and height indicate what space each sub-image gets. Normally, one neters the maximum height and width of any image.

        Parameters
        ----------
        image_list : list
            List of images

        width : int
            width (in pixels) of each subimage

        height : int
            height (in pixels) of each subimage

        nr_columns : int
            Number of columns

        percentage_space : float
            the percentage space between each subimage as a percentage of the width or height respectively
    """
    total_nr_images = len(image_list)  
    nr_rows = int(ceil(float(total_nr_images)/nr_columns))
    space = int(round(percentage_space*max(width, height)))
    new_image = PIL.Image.new("RGB", (round(width*nr_columns*(1+percentage_space)), round(height*nr_rows*(1+percentage_space))))

    for id_image in range(len(image_list)):
        image = image_list[id_image]
        id_x = id_image%nr_columns
        id_y = floor(id_image/nr_columns)
        new_image.paste(image, ((width+space)*id_x, (height+space)*id_y))
    return new_image

def get_image_list(image_file_name_list):
    """ Given a list with file names, it outputs a list of images that are loaded.

        Parameters
        ----------
        image_file_name_list : list
    """
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(os.path.join(IMGS_DIR, file_name)))
    return image_list

def get_max_size(image_list):
    """ Calculate the maximum height and width of all images in image_list. 

        Parameters
        ----------
        image_list : list
            List of images
    """
    width_max = 0
    height_max = 0
    for image in image_list:
        image_size = image.size
        width_max = max(image_size[0],width_max)
        height_max = max(image_size[1],height_max)
    return width_max, height_max

def main():
    """Concatentate all images in the image_file_name_list and saave them.

    Parameters
    """
    data_list = ['dsprites','chairs','celeba','mnist']
    model_list = ['VAE','betaH','betaB','factor','batchTC']

    image_file_name_list=[]
    for model_name in model_list:
        for data_name in data_list:
            image_file_name_list.append(model_name + '_' + data_name + '-prior-traversal.png')

    output_file_name = 'combined_image'
    
    nr_columns = len(data_list)

    image_list = get_image_list(image_file_name_list)
    width_max, height_max = get_max_size(image_list)
    percentage_space  = 0.03
    new_image = concatenate_images(image_list, width_max, height_max, nr_columns, percentage_space)
    
    new_image.save(os.path.join(IMGS_DIR, output_file_name + '.png'))
    return True

if __name__ == '__main__':
    main()
