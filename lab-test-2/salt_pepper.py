import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image

def main():
    img_path = './Salt_and_Pepper.png'
    print(img_path)

    img_gray = cv.imread(img_path, 0)
    # img_gray = 
    print(img_gray.shape)

    structuring_element = np.ones((5, 5))

    img_processed1 = cv.morphologyEx(img_gray, cv.MORPH_OPEN, structuring_element)
    img_processed2 = cv.morphologyEx(img_gray, cv.MORPH_CLOSE, structuring_element)
    img_processed3 = cv.dilate(img_processed1, structuring_element, iterations=2)
    # img_processed3 = img_processed3 - img_processed1 - img_processed2

    plt.imshow(img_processed1, cmap='gray')
    plt.show()

    plt.imshow(img_processed2, cmap='gray')
    plt.show()

    plt.imshow(img_processed3, cmap='gray')
    plt.show()

# def erosion(old_img, kernel):
#     row, col = old_img.shape
#     x, y = kernel.shape
#     new_img = np.zeros((row-x+1, col-y+1))
#     for i in range(row-x+1):
#         for j in range(col-y+1):
#             sum = np.sum(old_img[i:i+x, j:j+y] * kernel)
#             if (sum == np.sum(kernel * np.ones((x, y)))*1.0):
#                 new_img[i,j] = 1.0
#     return new_img

# def dilasion(old_img, kernel):
#     row, col = old_img.shape
#     x, y = kernel.shape
#     new_img = np.zeros((row-x+1, col-y+1))
#     for i in range(row-x+1):
#         for j in range(col-y+1):
#             if (np.sum(old_img[i:i+x, j:j+y] * kernel) >= 1.0):
#                 new_img[i,j] = 1.0
#     return new_img



if __name__ == "__main__":
    main()