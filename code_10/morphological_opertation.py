import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv

def main():
    img_path = './board-writing.jpg'
    img_gray = cv.imread(img_path, 0)
    
    # img_rgb = plt.imread(img_path)
    # print(img_rgb.shape)

    # img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    print(img_gray.shape)

    img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)[1]
    print(img_binary)

    kernel = np.ones((3, 3), dtype=np.uint8) 

    img_erosion1 = cv.erode(img_binary, kernel, iterations=1)
    img_erosion2 = erosion(img_binary, kernel)

    img_dilattion1 = cv.dilate(img_binary, kernel, iterations=1)
    img_dilattion2 = dilation(img_binary, kernel)

    img_opening1 = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel)
    img_opening2 = dilation(erosion(img_binary, kernel), kernel)

    img_closing1 = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)
    img_closing2 = erosion(dilation(img_binary, kernel), kernel)

    img_set = [ img_gray, img_binary, img_erosion1, img_erosion2, img_dilattion1, img_dilattion2, img_opening1, img_opening2, img_closing1, img_closing2]
    img_title = ['Grayscale', 'Binary', 'Erosion 1',  'Erosion 2', 'Dilation 1', 'Dilation 2', 'Opening 1', 'Opening 2', 'Closing 1', 'Closing 2']
    img_plot(img_set, img_title)

def img_plot(img, title):
    plt.figure(figsize=(15, 15))
    j = 0
    for i in range(int(len(img)/2)):
        plt.subplot(2, 5, i+1)
        plt.title(title[j])
        plt.imshow(img[j], cmap = 'gray')
        
        plt.subplot(2, 5, i+5+1)
        plt.title(title[j+1])
        plt.imshow(img[j+1], cmap = 'gray')
        j += 2

    plt.savefig('fig-1.jpg')
    plt.show()

def erosion(img_binary, kernel):
    r, c = img_binary.shape
    x, y = kernel.shape
    img_process = np.zeros((r-x-1, c-y-1))
    for i in range(r-x-1):
        for j in range(c-y-1):
            sum = np.sum(img_binary[i:i+x, j:j+y] * kernel) # only for fit, the pixel value will be 1
            check = np.sum(kernel * (255 * np.ones((3, 3))))
            if (sum == check):
                img_process[i][j] = 255
    return img_process

def dilation(img_binary, kernel):
    r, c = img_binary.shape
    x, y = kernel.shape
    img_process = np.zeros((r-x-1, c-y-1))
    for i in range(r-x-1):
        for j in range(c-y-1):
            sum = np.sum(img_binary[i:i+x, j:j+y] * kernel) 
            if (sum >= 255): # both for fit, hit the pixel value will be 1
                img_process[i][j] = 255
    return img_process

if __name__ == '__main__':
    main()