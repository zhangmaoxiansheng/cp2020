import numpy as np
import os
import cv2

def countTonemap(hdr, min_fraction=0.0005):
    counts, ranges = np.histogram(hdr, 256)
    min_count = min_fraction * hdr.size
    delta_range = ranges[1] - ranges[0]

    image = hdr.copy()
    for i in range(len(counts)):
        if counts[i] < min_count:
            image[image >= ranges[i + 1]] -= delta_range
            ranges -= delta_range
    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

folder = 'exposures'
img_file_list = list([os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1]=='.png'])

images = list([cv2.imread(f) for f in img_file_list])
average = images[0].astype(np.float)
for img in images[1:]:
    average += img

average /= len(images)

output = cv2.normalize(average, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite('output.png', output)

average_noise = images[-1]
average -= average_noise

output = cv2.normalize(average, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite('output2.png', output)
with open(os.path.join(folder, 'list.txt'),'r') as f:
    lines = f.readlines()
exposures = []
for line in lines:
    _, time = line.split()
    exposures.append(1. / float(time))

exposures = np.array(exposures).astype(np.float32)
calibration = cv2.createCalibrateDebevec()
response = calibration.process(images, exposures)

merge = cv2.createMergeDebevec()
hdr = merge.process(images, exposures, response)
cv2.imwrite('hdr_image.png', hdr)


#Non free module!
# durand = cv2.createTonemapDurand(gamma=2.5)
# ldr = durand.process(hdr)
ldr = countTonemap(hdr)
cv2.imwrite('durand_image.png', ldr * 255)