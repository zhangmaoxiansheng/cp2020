import numpy as np
import os
import cv2

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
cv2.imwrite('hdr_image.hdr', hdr)
output = cv2.normalize(hdr, None, 0, 255, cv2.NORM_MINMAX)
output = np.round(output)
output = output.astype(np.uint8)
cv2.imwrite('hdr_image.png', output)

durand = cv2.createTonemapDurand(gamma=2.5)
ldr = durand.process(hdr)
cv2.imwrite('durand_image.png', ldr * 255)