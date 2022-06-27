import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    path = './tower.jpg'
    print(path)

    rgb = plt.imread(path)
    print(rgb.shape)

    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY) + 50
    print(grayscale.shape)

    img_shift_left = np.copy(grayscale)
    img_shift_right = np.copy(grayscale)

    r, c = grayscale.shape

    img_shift_left -= 25
    # for i in range(r):
    #     for j in range(c):
    #         img_shift_left[i][j] -= 50
    
    img_shift_right += 25
    # for i in range(r):
    #     for j in range(c):
    #         img_shift_right[i][j] += 50

    img_range_check = np.copy(grayscale)
    
    for i in range(r):
        for j in range(c):
            if (img_range_check[i][j] <= 65):
                img_range_check[i][j] = 65
            elif (img_range_check[i][j] >= 212):
                img_range_check[i][j] = 212

    # print(img_range_check.shape)
    # print(img_range_check)

    gray_hist = cv.calcHist([grayscale], [0], None, [256], [0, 256])
    left_hist = cv.calcHist([img_shift_left], [0], None, [256], [0, 256])
    right_hist = cv.calcHist([img_shift_right], [0], None, [256], [0, 256])
    narrow_band_hist = cv.calcHist([img_range_check], [0], None, [256], [0, 256])

    img_set = [grayscale, img_shift_left, img_shift_right, img_range_check]
    title_set = ['Gray Image', 'Left shifted image', 'Right shifted image', 'Narrow band image']
    hist_set = [gray_hist, left_hist, right_hist, narrow_band_hist]

    plt.figure(figsize = (20, 20))
    j = 0
    for i in range(len(img_set)):
        j += 1
        plt.subplot(2, len(img_set), j)
        plt.title(title_set[i])
        plt.imshow(img_set[i], cmap = 'gray')

        plt.subplot(2, len(img_set), j+len(img_set))
        plt.title(title_set[i] + ' histogram')
        plt.plot(hist_set[i])

    plt.savefig('fig-1.jpg')
    plt.show()

if __name__ == '__main__':
    main()