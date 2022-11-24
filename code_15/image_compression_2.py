import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image 

src_img = cv2.imread('./trees_in_water.jpg',0)
cnt = 0
plt.imsave('./fig-' + str(cnt) + '.jpg', src_img, format='jpg', cmap='gray')

fft_img = np.fft.fft2(src_img)
fft_img_sort = np.sort(np.abs(fft_img.ravel()))
n = len(fft_img_sort)

print(src_img.shape,fft_img.shape , fft_img_sort.shape)

img_set = [src_img]
title_set = ['Source Image']

for keep in (0.75 , 0.5 , 0.1 , 0.05 , 0.01, 0.001 , 0.0001):

    thresh = fft_img_sort[int(np.floor(n * (1 - keep)))]
    ind = np.abs(fft_img) > thresh
    allow_pass = fft_img * ind
    ifft_img = np.fft.ifft2(allow_pass).real
    cnt += 1
    plt.imsave('./fig-' + str(cnt) + '.jpg', ifft_img, format='jpg', cmap='gray')
    img_set.append(ifft_img)
    title_set.append("Compressed (keep = {}%)".format(keep*100))

def plot_img(img_set , title_set):
    n = len(img_set)
    r , c = 2,4
    plt.figure(figsize=(20,20))
    for i in range(n):
        plt.subplot(r,c,i+1)
        plt.imshow(img_set[i],cmap='gray')
        plt.title(title_set[i])
    
    plt.savefig('output.jpg')
    plt.show()

plot_img(img_set,title_set)