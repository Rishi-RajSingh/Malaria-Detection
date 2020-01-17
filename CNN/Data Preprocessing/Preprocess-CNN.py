##### Input #####
# h x w x 3 images (int [0, 255])

##### Output #####
# 128 x 128 x 3 images (float [0, 1])
# Smooth image
# Split the Dataset into -
# 	Train set	- 22046
#	Eval set	- 2756
#	Test set	- 2756

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

uninfDir = './dataset/Uninfected/'
parDir = './dataset/Parasitized/'
classDirs = [uninfDir, parDir]

trainDir = './dataset/Train/'
evalDir = './dataset/Eval/'
testDir = './dataset/Test/'
splitDirs = [trainDir, evalDir, testDir]
splitSize = [0, 22046, 24802, 27588]

uninfImgs = os.listdir(uninfDir)
parImgs = os.listdir(parDir)
classImgs = [uninfImgs, parImgs]

cells = []
labels = []

resizeDim = (128, 128)
gaussFiltDim = (5, 5)
gaussFiltStdDev = 0		# 0 => std dev is calculated automatically according to the kernel size

for l in range(len(classImgs)):
	for imgName in classImgs[l]:
		img = cv2.imread(classDirs[l] + imgName)
		floatImg = img.astype(np.float32)/255.0
		resizeImg = cv2.resize(floatImg, resizeDim)
		smoothImg = cv2.GaussianBlur(resizeImg, gaussFiltDim, gaussFiltStdDev)
		cells.append(smoothImg)
		labels.append(l)

cells = np.array(cells)
labels = np.array(labels)

train_x, x, train_y, y = train_test_split(cells, labels, test_size = 0.2, random_state = 100)

eval_x, test_x, eval_y, test_y = train_test_split(x, y, test_size = 0.5, random_state = 100)

np.save(trainDir + 'cells.npy', train_x)
np.save(testDir + 'cells.npy', test_x)
np.save(evalDir + 'cells.npy', eval_x)
np.save(trainDir + 'labels.npy', train_y)
np.save(testDir + 'labels.npy', test_y)
np.save(evalDir + 'labels.npy', eval_y)