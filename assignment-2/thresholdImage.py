import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def main():
    img_path = 'trees_in_water_3.jpg'
    print(img_path)
    rgb = plt.imread(img_path)
    print(rgb.shape)

    T1 = 80
    T2 = 150
    c = 10
    p = 20
    epsilon = 1e-6

    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    r, c = grayscale.shape
    processed_img1 = np.zeros((r, c), dtype = np.uint8)
    for x in range(r):
        for y in range(c):
            if (grayscale[x][y] > T1 and grayscale[x][y] < T2):
                processed_img1[x][y] = 100
            else:
                processed_img1[x][y] = 10
    
    processed_img2 = grayscale
    for x in range(r):
        for y in range(c):
            if (processed_img2[x][y] > T1 and processed_img2[x][y] < T2):
                processed_img2[x][y] = 100

    processed_img3 = c * np.log(1 + grayscale)
    # processed_img3 = grayscale
    # for x in range(r):
    #     for y in range(c):
    #         processed_img3[x][y] = c*np.log(1+processed_img3[x][y])

    processed_img4 = c * ((r + grayscale) ** p)

    img_set = [rgb, grayscale, processed_img1, processed_img2, processed_img3, processed_img4]
    title_set = ['RGB', 'Grayscale', 'Processed Image 1', 'Processed Image 2', 'Processed Image 3', 'Processed Image 4']
    plot_img(img_set, title_set)

def plot_img(img_set, title_set):
    plt.figure(figsize = (20, 20))
    n = len(img_set)    
    for i in range(n):
        plt.subplot(2, 3, i+1)
        plt.title(title_set[i])
        if (i == 0):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
    plt.savefig('fig-1.jpg')    
    plt.show()

if __name__ == '__main__':
    main()