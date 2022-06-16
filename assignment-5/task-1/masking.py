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

    mask = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    mask[200:350, 50:300] = 255

    processed_img1 = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    processed_img1 = grayscale & mask
    
    binaryMask = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    binaryMask[200:350, 50:300] = 1

    processed_img2 = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    processed_img2 = grayscale & binaryMask

    img_set = [rgb, grayscale, processed_img1, processed_img2]
    img_title = ['RGB', 'Gray', 'Masked Image', 'Binary Masked Image']

    plt.figure(figsize = (20, 20))
    for i in range(len(img_set)):
        plt.subplot(2, 2, i+1)
        plt.title(img_title[i])
        if (i == 0):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')

    plt.savefig('mask-img.jpg')
    plt.show()

if __name__ == "__main__":
    main()