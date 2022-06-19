import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img_path = 'rose_2.jpg'
    print(img_path)
    rgb = plt.imread(img_path)
    print(rgb.shape)

    grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    print(grayscale.shape)
    
    avg_kernel = 1/9 * np.ones((3, 3))
    gaussian_kernel = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    processed_img = cv.filter2D(grayscale, -1, avg_kernel)

    noisy_img = np.copy(grayscale)
    for i in range(2000):
        x = np.random.randint(0, noisy_img.shape[0])
        y = np.random.randint(0, noisy_img.shape[1])
        noisy_img[x][y] = np.random.randint(0, 2) * 255
    
    processed_noisy_img1 = cv.filter2D(noisy_img, -1, avg_kernel)
    processed_noisy_img2 = cv.filter2D(noisy_img, -1, gaussian_kernel)

    # processed_noisy_img3 = np.zeros((noisy_img.shape[0], noisy_img.shape[1]))
    # for i in range(processed_noisy_img3.shape[0]-2):
    #     for j in range(processed_noisy_img3.shape[1]-2):
    #         processed_noisy_img3[i][j] = np.median(noisy_img[i:3+i, j:3+j])

    processed_noisy_img3 = cv.medianBlur(noisy_img, 3)

    img_set = [grayscale, processed_img, noisy_img, processed_noisy_img1, processed_noisy_img2, processed_noisy_img3]
    title_set = ['Gray', 'Filtered Image\n(Averaging)', 'Noisy Image\n(Salt & Pepper)', 'Filtered Noisy Image\n(Average Filter)', 'Filtered Noisy Image\n(Gaussian Filter)', 'Filtered Noisy Image\n(Median Filter)']
    
    plt.figure(figsize = (20, 20))
    for i in range(len(img_set)):
        plt.subplot(2, 3, i+1)
        plt.title(title_set[i])
        plt.imshow(img_set[i], cmap = 'gray')

    plt.savefig('noise.jpg')
    plt.show()

if __name__ == "__main__":
    main()