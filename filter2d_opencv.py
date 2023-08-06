import cv2
import numpy as np

image = cv2.imread('uepg_sm.jpg')

if image is None:
    print('Could not read image')
'''
# Identidade
kernel1 = np.array([[ 0, 0, 0],
                    [ 0, 1, 0],
                    [ 0, 0, 0]])
'''
# Detecção de borda
kernel1 = np.array([[-1,-1,-1],
                    [-1, 8,-1],
                    [-1,-1,-1]])
'''
# Nitidez
kernel1 = np.array([[ 0,-1, 0],
                    [-1, 5,-1],
                    [ 0,-1, 0]])

# Blur
kernel1 = np.array([[ 1, 1, 1],
                    [ 1, 1, 1],
                    [ 1, 1, 1]])*1/9

# Gaussian blur
kernel1 = np.array([[ 1, 4, 6, 4, 1],
                    [ 4,16,24,16, 4],
                    [ 6,24,36,24, 6],
                    [ 4,16,24,16, 4],
                    [ 1, 4, 6, 4, 1]])*1/256
'''

filtered = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

cv2.imwrite('filtered.jpg', filtered)
