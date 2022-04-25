import pandas as pd
import numpy as np
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
import os
from IPython.display import display
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import argparse
import logging
import pickle
import lmdb
import lz4framed
from itertools import tee

def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level


def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):
    patch_folder = os.path.join(patches_output_dir, slide_id)
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)
    else:
        return
    slide = OpenSlide(slide_path)

    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)
        mask, mask_level = get_mask(slide)
        mask = binary_dilation(mask, iterations=3)
        mask = binary_erosion(mask, iterations=3)
        np.save(os.path.join(patch_folder_mask, "mask.npy"), mask) 
    else:
        mask = np.load(os.path.join(mask_path, slide_id, 'mask.npy'))
        
    mask_level = len(slide.level_dimensions) - 1
    

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        #with open(os.path.join(patch_folder, 'loc.txt'), 'w') as loc:
        #loc.write("slide_id {0}\n".format(slide_id))
        #loc.write("id x y patch_level patch_size_read patch_size_output\n")

        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

        # handle slides with 40 magnification at base level
        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        resize_factor = resize_factor * args.dezoom_factor
        patch_size_resized = (int(resize_factor * patch_size[0]), int(resize_factor * patch_size[1]))
        i = 0

        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                       range(0, ymax, patch_size_resized[0])]
        np.random.seed(5)
        np.random.shuffle(indices)
        path_lmdb = os.path.join(patch_folder, slide_id + '.db')
        map_size = (max_patches_per_slide + 100) * 3 * patch_size[0] * patch_size[1]
        lmdb_connection = lmdb.open(path_lmdb, subdir=False,
                map_size=int(map_size), readonly=False, meminit=False, map_async=True)
        lmdb_txn = lmdb_connection.begin(write=True)
        for x, y in indices:
            # check if in background mask
            x_mask = int(x / ratio_x)
            y_mask = int(y / ratio_y)
            if mask[x_mask, y_mask] == 1:
                patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                try:
                    mask_patch = get_mask_image(np.array(patch))
                    mask_patch = binary_dilation(mask_patch, iterations=3)
                except Exception as e:
                    print("error with slide id {} patch {}".format(slide_id, i))
                    print(e)
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    if resize_factor != 1.0:
                        patch = patch.resize(patch_size)
                    #loc.write("{0} {1} {2} {3} {4} {5}\n".format(i, x, y, PATCH_LEVEL, patch_size_resized[0],
                    #                                                 patch_size_resized[1]))
                    #imsave(os.path.join(patch_folder, "{0}_patch_{1}.jpeg".format(slide_id, i)), np.array(patch))
                    img_idx: bytes = u"{}".format(i).encode('ascii')
                    img_name: bytes = u"{}".format("{0}_patch_{1}".format(slide_id, i))
                    patch = np.array(patch)
                    lmdb_txn.put(img_idx, serialize_and_compress((img_name, patch.tobytes(), patch.shape)))
                    i += 1
                    if i % 100 == 0:
                        lmdb_txn.commit()
                        lmdb_txn = lmdb_connection.begin(write=True)
            if i >= max_patches_per_slide:
                break

        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))
        lmdb_txn.commit()
        image_keys__list = [u'{}'.format(k).encode('ascii') for k in range(i)]
        with lmdb_connection.begin(write=True) as lmdb_txn:
            lmdb_txn.put(b'__keys__', serialize_and_compress(image_keys__list))
        lmdb_connection.sync()
        lmdb_connection.close()
    except Exception as e:
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)
        lmdb_connection.sync()
        lmdb_connection.close()


def serialize_and_compress(obj):
    return lz4framed.compress(pickle.dumps(obj))

def get_slide_id(slide_name):
    return slide_name.split('.')[0]+'.'+slide_name.split('.')[1]


def process(opts):
    # global lock
    slide_path, patch_size, patches_output_dir, mask_path, slide_id, max_patches_per_slide = opts
    extract_patches(slide_path, mask_path, patch_size,
                    patches_output_dir, slide_id, max_patches_per_slide)


parser = argparse.ArgumentParser(description='Generate patches from a given folder of images')
parser.add_argument('--wsi_path', required=True, metavar='WSI_PATH', type=str,
                    help='Path to the input directory of WSI files')
parser.add_argument('--patch_path', required=True, metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--mask_path', required=True, metavar='MASK_PATH', type=str,
                    help='Path to the  directory of numpy masks')
parser.add_argument('--patch_size', default=768, type=int, help='patch size, '
                                                                'default 768')
parser.add_argument('--max_patches_per_slide', default=2000, type=int)
parser.add_argument('--num_process', default=10, type=int,
                    help='number of mutli-process, default 10')
parser.add_argument('--dezoom_factor', default=1.0, type=float,
                    help='dezoom  factor, 1.0 means the images are taken at 20x magnification, 2.0 means the images are taken at 10x magnification')


if __name__ == '__main__':
    # count = Value('i', 0)
    # lock = Lock()

    args = parser.parse_args()
    slide_list = os.listdir(args.wsi_path)
    slide_list = [s for s in slide_list if s.endswith('.svs')]
    ##DEBUG

    #print("DEBUGING SMALL SLIDE LIST")
    #slide_list = ['GTEX-14A5I-0925.svs','GTEX-14A6H-0525.svs'
    #          ]
    
    opts = [
        (os.path.join(args.wsi_path, s), (args.patch_size, args.patch_size), args.patch_path, args.mask_path,
         get_slide_id(s), args.max_patches_per_slide) for
        (i, s) in enumerate(slide_list)]
    #pool = Pool(processes=args.num_process)
    #pool.map(process, opts)
    #process(opts)
    from tqdm import tqdm
    for opt in tqdm(opts):
        process(opt)
