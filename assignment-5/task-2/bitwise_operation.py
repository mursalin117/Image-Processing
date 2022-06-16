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

    bit1_img = np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    bit2_img = 2 * np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    bit3_img = 4 * np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    bit4_img = 8 *np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    bit5_img = 16 * np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    bit6_img = 32 * np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    bit7_img = 64 * np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    bit8_img = 128 * np.ones((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
        
    bit1_img = grayscale & bit1_img
    bit2_img = grayscale & bit2_img
    bit3_img = grayscale & bit3_img
    bit4_img = grayscale & bit4_img
    bit5_img = grayscale & bit5_img
    bit6_img = grayscale & bit6_img
    bit7_img = grayscale & bit7_img
    bit8_img = grayscale & bit8_img
    
    # bit2_img = np.zeros((grayscale.shape[0], grayscale.shape[1]), dtype = np.uint8)
    # for i in range(grayscale.shape[0]):
    #     for j in range(grayscale.shape[1]):
    #         bit2_img[i][j] = grayscale[i][j] & 1
    bit_img = [rgb, grayscale, bit1_img, bit2_img, bit3_img, bit4_img, bit5_img, bit6_img, bit7_img, bit8_img]
    bit_title = ['RGB', 'Gray','LSB', '2nd LSB', '3rd LSB', '4th LSB', '4th MSB', '3rd MSB', '2nd MSB', 'MSB'] 
    
    plt.figure(figsize = (20, 20))
    for i in range(len(bit_img)):
        plt.subplot(2, 5, i+1)
        plt.title(bit_title[i])
        if (i == 0):
            plt.imshow(bit_img[i])
        else :
            plt.imshow(bit_img[i], cmap = 'gray')

    plt.savefig('bit-wise-image.jpg')
    plt.show()


if __name__ == "__main__":
    main()