import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def main():
    img_path = './table.jpg'
    rgb = plt.imread(img_path)
    print(rgb.shape)

    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    print(grayscale.shape)

    horizontalKernel = np.array([[3, 10, 3],[0, 0, 0], [-3, -10, -3]])
    verticalKernel = np.array([[3, 0, -3],[10, 0, -10], [3, 0, -3]])
    sharpenKernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
    identityKernel = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])
    boxBlurKernel = 1/9 * np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
    gaussianBlurKernel = 1/16 * np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])
    outlineKernelKernel = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])

    processed_img1 = cv.filter2D(grayscale, -1, horizontalKernel)
    processed_img2 = cv.filter2D(grayscale, -1, verticalKernel)
    processed_img3 = cv.filter2D(grayscale, -1, sharpenKernel)
    processed_img4 = cv.filter2D(grayscale, -1, identityKernel)
    processed_img5 = cv.filter2D(grayscale, -1, boxBlurKernel)
    # processed_img6 = cv.filter2D(grayscale, -1, gaussianBlurKernel)
    processed_img7 = cv.filter2D(grayscale, -1, outlineKernelKernel)

    img_set = [rgb, grayscale, processed_img1, processed_img2, processed_img3, processed_img4, processed_img5, processed_img7]#, processed_img6]
    title_set = ['RGB', 'Grayscale', 'Horizontal', 'Vertical', 'Sharpen', 'Identity', 'Box Blur', 'Outline']#, 'Gaussian Blur']
    plot_img(img_set, title_set)

def plot_img(img_set, title_set):
    n = len(img_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        plt.subplot(2, 4, i+1)
        plt.title(title_set[i])
        if (i == 0):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
    plt.savefig('fig-1.jpg')
    plt.show()


if __name__ == '__main__':
    main()