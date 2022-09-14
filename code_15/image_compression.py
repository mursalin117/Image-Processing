import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img_path = './trees_in_water.jpg'
    print(img_path)

    img_rgb = plt.imread(img_path)
    print(img_rgb.shape)

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    print(img_gray.shape)

    ft = np.fft.fft2(img_gray)
    ftSort = np.sort(np.abs(ft.reshape(-1))) # sort by highest magnitude

    # Zero out all the small co-efficients and inverse transformation
    for keep in (0.1, 0.05, 0.01, 0.002):
        thresh = ftSort[int(np.floor((1-keep)*len(ftSort)))]
        ind = np.abs(ft) > thresh           # find small indices
        Atlow = ft * ind                    # threshold small indices
        ftlow = np.fft.ifft2(Atlow).real    # compressed image
        # plt.figure()
        plt.imshow(ftlow, cmap='gray')
        plt.show()
        print(ftlow.shape)

if __name__ == '__main__':
    main()