import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img_path = './tower.jpg'
    img_gray = cv.imread(img_path, 0)
    print(img_gray.shape)

    img1 = np.copy(img_gray)
    img2 = np.copy(img_gray)

    hist = cv.calcHist(img1, [0], None, [256], [0, 256])
    
    img_hist_equalized1 = cv.equalizeHist(img1)
    new_hist1 = cv.calcHist(img_hist_equalized1, [0], None, [256], [0, 256]) 

    img_hist_equalized2 = equalizeHistogram(img2)
    # print(img_hist_equalized2.shape)
    # print(img_hist_equalized2)
    new_hist2 = cv.calcHist(img_hist_equalized2, [0], None, [256], [0, 256])

    set1 = [img1, hist, img_hist_equalized1, new_hist1]
    set2 = [img2, hist, img_hist_equalized2, new_hist2]
    title = ['Original\nImage', 'Original\nHistogram', 'Equalized\nImage', 'Equalized\nHistogram']

    plt.figure(figsize=(15, 15))
    for i in range(len(set1)):
        plt.subplot(2, 4, i+1)
        plt.title(title[i])
        if (i % 2 == 0):
            plt.imshow(set1[i], cmap='gray')
        else: 
            plt.plot(set1[i])

        plt.subplot(2, 4, i+1+4)
        plt.title(title[i])
        if (i % 2 == 0):
            plt.imshow(set2[i], cmap='gray')
        else: 
            plt.plot(set2[i])
    
    plt.savefig('fig-1.jpg')
    plt.show()

def equalizeHistogram(img_old):
    
    r, c = img_old.shape 
    frequency = np.arange(0, 256)
    frequency[:] = 0

    # frequency count
    for i in range(r):
        for j in range(c):
            frequency[img_old[i][j]] += 1
    
    # total no of frequency
    total = r*c
    
    # probability of the frequencies
    pdf = []
    for i in range(256):
        pdf.append(frequency[i]/total)
    
    # probability of cumulative frequencies
    cdf = []
    cf = 0
    for i in range(256):
        cf += frequency[i]
        cdf.append(cf/total)
    
    # new pixel values
    new_pixel = np.arange(0, 256)
    for i in range(256):
        new_pixel[i] = round(cdf[i] * 255)

    # assigning new values to new image (equalized image)
    img_new = np.zeros((r, c), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            img_new[i][j] = new_pixel[img_old[i][j]]

    return img_new

if __name__ == '__main__':
    main()