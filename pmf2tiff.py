# inspired from: https://github.com/wu258/Middlebury-Stereo-dataset-pfm-to-png

from pathlib import Path
from PIL import Image
import numpy as np
import re
import cv2
import os

def get_pfms_path(path, result):
    for filename in os.listdir(path):
        file_path = path + filename
        if os.path.isdir(file_path):
            get_pfms_path(file_path + "/" , result)
        else:
            file_type = os.path.splitext(file_path)[-1]
            if ".pfm" == file_type:
                result.append(file_path)
    return result

def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")
        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # little endian
            scale = -scale
        else:
            endian = '>' # big endian
        dispariy = np.fromfile(pfm_file, endian + 'f')
        dispariy = np.reshape(dispariy, newshape=(height, width, channels))
        dispariy = np.flipud(dispariy)
    return dispariy

def main():
    path_set = []
    path_set = get_pfms_path('./data/', path_set)
    for item in path_set:
        dispariy = read_pfm(item)
        dispariy = np.reshape(dispariy, (len(dispariy), len(dispariy[0])))
        file_prefix = os.path.splitext(item)[0]
        dispariy = Image.fromarray(dispariy)
        dispariy.save(file_prefix + ".tiff")
        

if __name__ == '__main__':
    main()
