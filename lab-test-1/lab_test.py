import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    cnt = 1
    # task -1
    img_path = './cat-1.jpg'
    print(img_path)

    img_rgb = plt.imread(img_path)
    print(img_rgb.shape)

    bit6_plane_img = (1<<6) & img_rgb
    bit7_plane_img = (1<<7) & img_rgb

    # img_plot([bit6_plane_img, bit7_plane_img], ['Bit -6 Plane Image', 'Bit -7 Plane Image'], cnt)

    # task-2
    img_gray = rgb_to_gray(img_rgb)
    print(img_gray.shape)

    cnt += 1
    img_plot([img_rgb, img_gray], ['RGB', 'Gray'], cnt)

    img_binary = gray_to_binary(img_gray, 127)
    print(img_binary.shape)

    cnt += 1
    img_plot([img_gray, img_binary], ['Gray', 'Binary'], cnt)

    gray_hist = histogram(img_gray)
    # print(gray_hist)
    cnt += 1
    plt.plot(gray_hist)
    plt.savefig('task' + str(cnt) + '.jpg')
    plt.show()

    # task-3
    kernel1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # task-4
    first_img = convolution(img_gray, kernel1)
    second_img = convolution(img_gray, kernel2)

    cnt += 1
    img_plot([first_img, second_img], ['Kernel1', 'Kernel2'], cnt)

def convolution(img, kernel):
    r, c = img.shape
    x, y = kernel.shape

    img_new = np.zeros((r-x-1, c-y-1))
    for i in range(r-x-1):
        for j in range(c-y-1):
            check = np.sum(img[i:3+i, j:j+3] * kernel)
            if (check > 255):
                img_new[i][j] = 255
            elif (check < 0):
                img_new[i][j] = 0
            else: 
                img_new[i][j] = check
    
    return img_new

def histogram(img):
    frequency = np.arange(0, 255)
    # frequency = 0
    for i in range(255):
        frequency[i] = 0
        
    r, c = img.shape
    for i in range(r):
        for j in range(c):
                frequency[int(img[i][j])] += 1
    return frequency    

def gray_to_binary(img, val):
    r, c = img.shape
    img_new = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            if(img[i][j] > val):
                img_new[i][j] = 255
    return img_new
    


def rgb_to_gray(img):
    r = img.shape[0]
    c = img.shape[1]
    print(r, c)
    red = img[:, :, 0]
    print(red.shape)
    green = img[:, :, 1]
    print(green.shape)
    blue = img[:, :, 2]
    print(blue.shape)
    img_new = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            img_new[i][j] = 0.14 * red[i][j] + 0.74 * green[i][j] + 0.12 * blue[i][j]
    return img_new 

def img_plot(img_set, img_title, cnt):
    plt.figure(figsize = (20, 20))
    for i in range(len(img_set)):
        plt.subplot(1, 2, i+1)
        plt.title(img_title[i])
        plt.imshow(img_set[i], cmap='gray')
    
    plt.savefig("task" + str(cnt) + '.jpg')
    plt.show()

if __name__ == '__main__':
    main()