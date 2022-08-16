import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img_path = "./writing.jpg"
    print(img_path)

    img_rgb = plt.imread(img_path)
    print(img_rgb.shape)

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    print(img_gray.shape)

    img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)[1]
    print(img_binary)

    kernel1 = np.ones((3, 3))
    kernel2 = np.array([[1, 1, 1],[2, 2, 2], [3, 3, 3]])
    kernel3 = np.array([[3, 3, 3],[2, 2, 2], [1, 1, 1]])

    img_erosion1 = cv.erode(img_binary, kernel1, iterations=1)
    img_erosion2 = cv.erode(img_binary, kernel2, iterations=1)
    img_erosion3 = cv.erode(img_binary, kernel3, iterations=1)

    img_dilattion1 = cv.dilate(img_binary, kernel1, iterations=1)
    img_dilattion2 = cv.dilate(img_binary, kernel2, iterations=1)
    img_dilattion3 = cv.dilate(img_binary, kernel3, iterations=1)

    img_opening1 = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel1)
    img_opening2 = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel2)
    img_opening3 = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel3)
    
    img_closing1 = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel1)
    img_closing2 = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel2)
    img_closing3 = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel3)

    img_set1 = [img_rgb, img_gray, img_binary, img_erosion1, img_dilattion1, img_opening1, img_closing1]
    img_set2 = [img_rgb, img_gray, img_binary, img_erosion2, img_dilattion2, img_opening2, img_closing2]
    img_set3 = [img_rgb, img_gray, img_binary, img_erosion3, img_dilattion3, img_opening3, img_opening3]
    img_all = [img_set1, img_set2, img_set3]
    img_title = ['RGB', 'Gray', 'Binary', 'Erosion', 'Dilation', 'Opening', 'Closing']
    
    for i in range(3):
        img_plot(img_all[i], img_title, i+1)

def img_plot(img_set, title_set, cnt):
    
    plt.figure(figsize=(30, 30))
    for i in range(len(img_set)):
        plt.subplot(2, 4, i+1)
        plt.title(title_set[i])
        if (i == 0):
            plt.imshow(img_set[i])
        # elif (i == 1):
        #     plt.imshow(img_set[i], cmap='gray')
        else: 
            plt.imshow(img_set[i], cmap='gray')
    plt.savefig('fig' + str(cnt) + '.jpg')
    plt.show()

if __name__ == "__main__":
    main()