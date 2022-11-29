import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import os

def gt_stereo_is_valid(value):
    return value != float("inf")

def output_stereo_is_valid(value):
    return value >= 0

def eval(algorithm, numDisparities, blockSize, path):
    if algorithm == "StereoBM":
        stereo = cv.StereoBM_create(numDisparities, blockSize)
    elif algorithm == "StereoSGBM":
        blockSize = 11
        # min_disp = -128
        # max_disp = 128
        # # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # # In the current implementation, this parameter must be divisible by 16.
        # num_disp = max_disp - min_disp
        # # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # # Normally, a value within the 5-15 range is good enough
        # uniquenessRatio = 5
        # # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        # speckleWindowSize = 200
        # # Maximum disparity variation within each connected component.
        # # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # # Normally, 1 or 2 is good enough.
        speckleRange = 2
        # disp12MaxDiff = 0

        stereo = cv.StereoSGBM_create(
            # minDisparity=min_disp,
            numDisparities=numDisparities,
            blockSize=blockSize,
            # uniquenessRatio=uniquenessRatio,
            # speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            # disp12MaxDiff=disp12MaxDiff,
            # P1=8 * 1 * blockSize * blockSize,
            # P2=32 * 1 * blockSize * blockSize,
        )
        # stereo = cv.StereoSGBM_create(numDisparities, blockSize)
    else:
        raise Exception("Algorithm must be either StereoBM or StereoSGBM")

    num_dataset = 0
    latency_avg_ms = 0.0
    output_valid_ratio_avg = 0.0
    output_error_ratio_avg = 0.0

    for folder_name in os.listdir(path):
        num_dataset += 1
        folder_name = path + folder_name
        img0 = cv.imread(folder_name + '/im0.png', 0)
        img1 = cv.imread(folder_name + '/im1.png', 0)
        t1 = time.perf_counter()
        output_disparity = stereo.compute(img0, img1)
        t2 = time.perf_counter()
        # latency in ms
        latency_avg_ms += (t2 - t1) * 1000

        output_disparity = output_disparity.astype(np.float32) / 16.0

        # plt.imshow(output_disparity,'gray')
        # plt.show()

        # cv.imwrite('./data/artroom1/disparity.tiff', output_disparity)

        gt = cv.imread(folder_name + '/disp0.tiff', cv.IMREAD_UNCHANGED)

        # evaluate accuracy
        assert output_disparity.shape == gt.shape
        num_gt_valid = 0;
        output_valid_ratio = 0.0
        output_error_ratio = 0.0

        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                if (gt_stereo_is_valid(gt[i][j])):
                    num_gt_valid += 1
                    if (output_stereo_is_valid(output_disparity[i][j])):
                        output_valid_ratio += 1
                        output_error_ratio += abs((output_disparity[i][j] - gt[i][j]) / gt[i][j])
        output_valid_ratio /= num_gt_valid
        output_error_ratio /= num_gt_valid

        output_valid_ratio_avg += output_valid_ratio
        output_error_ratio_avg += output_error_ratio

    latency_avg_ms /= num_dataset
    output_valid_ratio_avg /= num_dataset
    output_error_ratio_avg /= num_dataset

    return latency_avg_ms, output_valid_ratio_avg, output_error_ratio_avg

if __name__ == '__main__':
    latency_avg_ms, output_valid_ratio_avg, output_error_ratio_avg = eval("StereoBM", 160, 105, './data/')
    print("latency_avg_ms:", latency_avg_ms)
    print("output_valid_ratio_avg: ", output_valid_ratio_avg)
    print("output_error_ratio_avg:", output_error_ratio_avg)
