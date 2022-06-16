import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img_path = './trees_in_water_3.jpg'
    print(img_path)
    rgb = plt.imread(img_path)
    print(rgb.shape)

    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    print(grayscale.shape)

    laplaceKernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplaceKernel2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplaceKernel3 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplaceKernel4 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    sobelKernel1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobelKernel2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    processed_img1 = cv.filter2D(grayscale, -1, laplaceKernel1)
    processed_img2 = cv.filter2D(grayscale, -1, laplaceKernel2)
    processed_img3 = cv.filter2D(grayscale, -1, laplaceKernel3)
    processed_img4 = cv.filter2D(grayscale, -1, laplaceKernel4)

    processed_img5 = cv.filter2D(grayscale, -1, sobelKernel1)
    processed_img6 = cv.filter2D(grayscale, -1, sobelKernel2)

    set_img = [rgb, grayscale, processed_img1, processed_img2, processed_img3, processed_img4, processed_img5, processed_img6]
    set_title = ['RGB', 'Gray', 'Laplace Kernel1', 'Laplace Kernel2', 'Laplace Kernel3', 'Laplace Kernel4', 'Sobel Kernel1', 'Sobel Kernel2']

    plt.figure(figsize = (20, 20))
    for i in range(len(set_img)):
        plt.subplot(2, 4, i+1)
        plt.title(set_title[i])
        if (i == 0):
            plt.imshow(set_img[i])
        else:
            plt.imshow(set_img[i], cmap = 'gray')
    plt.savefig('Laplace-&-Sobel-Filtering.jpg')
    plt.show()

if __name__ == "__main__":
    main()