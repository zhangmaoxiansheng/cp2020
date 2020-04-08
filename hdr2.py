import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

'''
argument:
Z(i,j) is the pixel values of pixel location i in image j 
B(j) is the log delta t 
l is the smooth term
w(z) is the weighting function
Zmin = 0
Zmax = 255
'''
def gsolve(Z, B,l ,w):
    n = 256
    
    A = np.zeros((Z.shape[0]*Z.shape[1]+n+1, n+Z.shape[0]))
    b = np.zeros((A.shape[0],1))

    k=0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[Z[i,j]+1]
            A[k, Z[i,j]+1] = wij
            A[k,n+i] = -wij
            b[k,0] = wij * B[j]
            k = k+1

    A[k,128] = 1
    k = k+1
    for i in range(n-2):
        A[k, i] = l * w[i+1]
        A[k, i+1] = -2 * l * w[i+1]
        A[k, i+2] = l * w[i+1]
        k = k + 1 

    x, _, _, _ = np.linalg.lstsq(A, b)
    g = x[0:n]
    lE = x[n:]
    return g, lE

def compute_hdr_map(images, g_red, g_green, g_blue, weights, ln_dt):
    img_num = len(images)
    height, width, channel = images[1].shape
    numerator = np.zeros((height, width, channel))
    denominator = np.zeros((height, width, channel))
    curr_num = np.zeros((height, width, channel))
    curr_weight = np.zeros((height, width, channel))
    for i in range(img_num):
        curr_image = images[i].astype(np.float32)
        curr_red = curr_image[:,:,0]
        curr_green = curr_image[:,:,1]
        curr_blue = curr_image[:,:,2]
        for x in np.nditer(curr_weight, op_flags=['readwrite']):
            x[...] = weights[int(x)]
        for x in np.nditer(curr_red, op_flags=['readwrite']):
            x[...] = g_red[int(x)]
        for x in np.nditer(curr_green, op_flags=['readwrite']):
            x[...] = g_green[int(x)]
        for x in np.nditer(curr_blue, op_flags=['readwrite']):
            x[...] = g_blue[int(x)]
        curr_num[:,:,0] = curr_weight[:,:,0] * (curr_red - ln_dt[i])
        curr_num[:,:,1] = curr_weight[:,:,1] * (curr_green - ln_dt[i])
        curr_num[:,:,2] = curr_weight[:,:,2] * (curr_blue - ln_dt[i])

        numerator = numerator + curr_num
        denominator = denominator + curr_weight
    
    ln_hdr_map = numerator / denominator
    hdr_map = np.exp(ln_hdr_map)
    return hdr_map

def plot_radiance_map(rmap):
    rmap = rmap / np.max(rmap)
    rmap[np.where(rmap < 0)] = 0
    rmap[np.where(rmap > 1)] = 1
    h = figure
    plt.imshow(h, cmap='plasma')
    plt.show()

def compute_weight():
    weigths = [min(w,256-w) for w in range(1,257)]
    weights = np.asarray(weigths)
    return weigths

def read_img_time(folder='exposures',extention='.png'):
    img_file_list = list([os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1]=='.png'])
    images = list([cv2.imread(f) for f in img_file_list])
    with open(os.path.join(folder, 'list.txt'),'r') as f:
        lines = f.readlines()
    exposures = []
    for line in lines:
        _, time = line.split()
        exposures.append(1. / float(time))
    exposures = np.array(exposures).astype(np.float32)
    return images, exposures

def sample_rgb_images(images):
    total_sample_num = len(images)
    # num_sample = 255 / (total_sample_num-1) * 2
    # num_sample = np.round(255 / num_sample)
    # num_sample = int(num_sample)
    num_sample = 50

    img_pixels = images[1].shape[1] * images[1].shape[2]
    step = int(img_pixels / num_sample)
    sample_indices = list(range(0,step,img_pixels))

    z_red = np.zeros((num_sample, total_sample_num), dtype = np.int)
    z_green = np.zeros((num_sample, total_sample_num), dtype = np.int)
    z_blue = np.zeros((num_sample, total_sample_num), dtype = np.int)

    for i in range(total_sample_num):
        sample_red, sample_green, sample_blue = sample_exposure(images[i], sample_indices)
        z_red[:,i] = sample_red.reshape(-1,1)
        z_green[:,i] = sample_green.reshape(-1,1)
        z_blue[:,i] = sample_blue.reshape(-1,1)
    return z_red, z_green, z_blue


def sample_exposure(image, sample_indices):
    blue_img = image[:,:,0]
    blue_img = blue_img.flatten()
    green_img = image[:,:,1]
    green_img = green_img.flatten()
    red_img = image[:,:,2]
    red_img = red_img.flatten()

    sample_red = red_img[sample_indices]
    sample_green = green_img[sample_indices]
    sample_blue = blue_img[sample_indices]
    return sample_red, sample_green, sample_blue




if __name__ == "__main__":
    images, exposure_times = read_img_time()
    ln_dt = np.log(exposure_times)
    z_red, z_green, z_blue = sample_rgb_images(images)
    weights = compute_weight()
    l = 50
    g_red,_ = gsolve(z_red, ln_dt, l, weights)
    g_green,_ = gsolve(z_green, ln_dt, l, weights)
    g_blue,_ = gsolve(z_blue, ln_dt, l, weights)
    hdr_map = compute_hdr_map(images, g_red, g_green, g_blue, weights, ln_dt)
    cv2.imwrite('hdr_map.png', hdr_map)
    print('finish')

