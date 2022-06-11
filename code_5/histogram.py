import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
        
def main():
    img_path = './table.jpg'
    print(img_path)
    rgb = plt.imread(img_path)
    print(rgb.shape)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    histRed = cv.calcHist([rgb], [0], None, [256], [0, 256])
    histGreen = cv.calcHist([rgb], [1], None, [256], [0, 256])
    histBlue = cv.calcHist([rgb], [2], None, [256], [0, 256])

    histRed2 = histogram(red)
    histGreen2 = histogram(green)
    histBlue2 = histogram(blue)

    set_histogram = [histRed, histGreen, histBlue, [], histRed2, histGreen2, histBlue2, []]
    set_title = ['Red', 'Green', 'Blue', 'RGB', 'Red', 'Green', 'Blue', 'RGB']
    
    plt.figure(figsize = (20, 20))
    for i in range(len(set_histogram)):
        plt.subplot(2, 4, i+1)
        plt.title(set_title[i])
        if (i  == 3 or i == 7):
            plt.plot(set_histogram[i-3], 'r')
            plt.plot(set_histogram[i-2], 'g')
            plt.plot(set_histogram[i-1], 'b')
        else:
            plt.plot(set_histogram[i])
    plt.savefig('histogram.jpg')
    plt.show()

def histogram(arr):
    frequencyCount = np.arange(0, 256)
    frequencyCount[:] = 0
    # print(frequencyCount)
    # print(arr) 

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if (arr[i][j] != 0):
                frequencyCount[arr[i][j]] += 1
    
    return frequencyCount     

if __name__ == '__main__':
    main()