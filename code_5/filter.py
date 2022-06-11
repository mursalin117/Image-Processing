import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def main():
    img_path = './table.jpg'
    print(img_path)
    rgb = plt.imread(img_path)
    print(rgb.shape)

    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    kernel1 = np.ones((3, 3), dtype = np.uint8)
    kernel2 = np.array([[3, 10, 3],[0, 0, 0], [-3, -10, -3]])

    processed_img11 = cv.filter2D(grayscale, -1, kernel1)
    processed_img12 = convolution(grayscale, kernel1)

    processed_img21 = cv.filter2D(grayscale, -1, kernel2)
    processed_img22 = convolution(grayscale, kernel2)

    img_set = [processed_img11, processed_img21, processed_img12, processed_img22]
    img_title = ['Blur1', 'Horizontal1', 'Blur2', 'Horizontal2']

    plt.figure(figsize = (20, 20))
    for i in range(len(img_set)):
        plt.subplot(2, 2, i+1)
        plt.title(img_title[i])
        plt.imshow(img_set[i], cmap = 'gray')

    plt.savefig('filter.jpg')
    plt.show()

def convolution(img, kernel):
    processed_img = np.zeros((img.shape[0]-2, img.shape[1]-2))
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]-2):
            temp = np.sum(img[i:3+i, j:3+j] * kernel)
            if (temp > 255):
                processed_img[i][j] = 255
            elif (temp < 0):
                processed_img[i][j] = 0
            else: 
                processed_img[i][j] = temp
    
    return processed_img

if __name__ == '__main__':
    main()