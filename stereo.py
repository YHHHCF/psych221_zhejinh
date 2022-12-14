import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import time
import os

############################ experiment parameters ############################
# 1 for debug (to run on 1 data), 0 for non-debug (to run on all 24 data)
debug = 1

# 1 to run and save results, 0 to load and plot results
compute = 1

data_path = './data/' # input dataset
result_path = './result/SGBM' # for either saving or loading
algorithm = 'StereoSGBM' # the algorithm to be evaluated
###############################################################################

def gt_stereo_is_valid(value):
    return value != float("inf")

def output_stereo_is_valid(value):
    return value >= 0

def eval(algorithm, numDisparities, blockSize, path):
    if algorithm == "StereoBM":
        stereo = cv.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    elif algorithm == "StereoSGBM":
        stereo = cv.StereoSGBM_create(numDisparities=numDisparities, blockSize=blockSize)
    else:
        raise Exception("Algorithm must be either StereoBM or StereoSGBM")

    num_dataset = 0
    latency_avg_ms = 0.0
    output_valid_ratio_avg = 0.0
    output_error_ratio_avg = 0.0

    for folder_name in os.listdir(path):
        if (folder_name == ".DS_Store"):
            continue

        num_dataset += 1
        folder_name = path + folder_name
        img0 = cv.imread(folder_name + '/im0.png', 0) # left
        img1 = cv.imread(folder_name + '/im1.png', 0) # right
        t1 = time.perf_counter()
        output_disparity = stereo.compute(img0, img1) # disp for left

        t2 = time.perf_counter()
        # latency in ms
        latency_avg_ms += (t2 - t1) * 1000 / 2

        # convert from 4-bit fixed point to floating point
        output_disparity = output_disparity.astype(np.float32) / 16.0

        gt_disp = cv.imread(folder_name + '/disp0.tiff', cv.IMREAD_UNCHANGED)

        # evaluate accuracy
        assert output_disparity.shape == gt_disp.shape

        num_gt_valid = 0;
        output_valid_ratio = 0.0
        output_error_ratio = 0.0

        for i in range(gt_disp.shape[0]):
            for j in range(gt_disp.shape[1]):
                if (gt_stereo_is_valid(gt_disp[i][j])):
                    num_gt_valid += 1
                    if (output_stereo_is_valid(output_disparity[i][j])):
                        output_valid_ratio += 1
                        output_error_ratio += abs((output_disparity[i][j] - gt_disp[i][j]) / gt_disp[i][j])
        output_valid_ratio /= num_gt_valid
        output_error_ratio /= num_gt_valid

        output_valid_ratio_avg += output_valid_ratio
        output_error_ratio_avg += output_error_ratio

        # if debug, process and show result on only 1 data and exit
        if debug:
            cv.imwrite(folder_name + '/disparity_' + algorithm + '_' + str(numDisparities) + '_' + str(blockSize) + '.tiff', output_disparity)
            break;

    latency_avg_ms /= num_dataset
    output_valid_ratio_avg /= num_dataset
    output_error_ratio_avg /= num_dataset

    return latency_avg_ms, output_valid_ratio_avg, output_error_ratio_avg

if __name__ == '__main__':
    if compute:
        # compute and save the results
        numDisparitiesList = np.array([16, 128, 256])
        blockSizeList = np.array([5, 21, 41])

        latency_result = np.zeros((blockSizeList.size, numDisparitiesList.size))
        recall_result = np.zeros((blockSizeList.size, numDisparitiesList.size))
        error_result = np.zeros((blockSizeList.size, numDisparitiesList.size))

        for i in range(numDisparitiesList.size):
            numDisparities = numDisparitiesList[i]
            for j in range(blockSizeList.size):
                blockSize = blockSizeList[j]
                print(f">>> {algorithm} experiment for numDisparities = {numDisparities} and blockSize = {blockSize}")
                latency_avg_ms, output_valid_ratio_avg, output_error_ratio_avg = eval(algorithm, numDisparities, blockSize, data_path)
                print(f"average latency: {latency_avg_ms:.1f} ms")
                print(f"average relative error: {output_error_ratio_avg * 100:.2f}%")
                print(f"average recall: {output_valid_ratio_avg * 100:.2f}%")
                latency_result[j][i] = latency_avg_ms
                recall_result[j][i] = output_valid_ratio_avg * 100
                error_result[j][i] = output_error_ratio_avg * 100

        np.save(result_path + 'numDisparitiesList.npy', numDisparitiesList)
        np.save(result_path + 'blockSizeList.npy', blockSizeList)
        np.save(result_path + 'latency_result.npy', latency_result)
        np.save(result_path + 'recall_result.npy', recall_result)
        np.save(result_path + 'error_result.npy', error_result)
    else:
        #load and plot the results
        numDisparitiesList = np.load(result_path + 'numDisparitiesList.npy')
        blockSizeList = np.load(result_path + 'blockSizeList.npy')
        latency_result = np.load(result_path + 'latency_result.npy')
        recall_result = np.load(result_path + 'recall_result.npy')
        error_result = np.load(result_path + 'error_result.npy')

        print("Loading succeed")

        fig = plt.figure()
        X, Y = np.meshgrid(numDisparitiesList, blockSizeList)

        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot_surface(X, Y, latency_result)
        ax1.set_xlabel('numDisparities')
        ax1.set_ylabel('blockSize')
        ax1.set_zlabel('latency (ms)')

        ax2 = fig.add_subplot(222, projection='3d')
        ax2.plot_surface(X, Y, error_result)
        ax2.set_xlabel('numDisparities')
        ax2.set_ylabel('blockSize')
        ax2.set_zlabel('relative error (%)')

        ax3 = fig.add_subplot(223, projection='3d')
        ax3.plot_surface(X, Y, recall_result)
        ax3.set_xlabel('numDisparities')
        ax3.set_ylabel('blockSize')
        ax3.set_zlabel('recall (%)')

        plt.show()
