import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv

def main():
    # Load image
    img_path = './tower.jpg'
    img_rgb = cv.imread(img_path)

    # Convert images
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    # Perform Fast Fourier Transformation for 2D signal, i.e., image
    img_ft = np.fft.fft2(img_gray)
    img_ft_centered = np.fft.fftshift(img_ft)
    magnitude_spectrum = 100 * np.log(np.abs(img_ft))
    centered_magnitude_spectrum = 100 * np.log(np.abs(img_ft_centered))

    print(img_gray.shape, img_ft.shape, img_ft_centered.shape)
    print(img_gray.max(), img_gray.min(), img_ft.max(), img_ft.min(), img_ft_centered.max(), img_ft_centered.min())

    # Build four different filters
    r, c = img_gray.shape
    kernel1 = np.zeros((r, c))
    kernel1[100:110, :] = 255
    kernel2 = np.zeros((r, c))
    kernel2[:, int(c/2-10):int(c/2+10)] = 255
    kernel3 = np.zeros((r, c))
    kernel3[int(r/4):int(r*3/4), int(c/4):int(c*3/4)] = 255
    kernel4 = np.ones((r, c))

    # Apply the filters
    img_filtered1 = img_ft_centered * kernel1
    img_filtered1_ifft = np.abs(np.fft.ifft2(img_filtered1))

    img_filtered2 = img_ft_centered * kernel2
    img_filtered2_ifft = np.abs(np.fft.ifft2(img_filtered2))

    img_filtered3 = img_ft_centered * kernel3
    img_filtered3_ifft = np.abs(np.fft.ifft2(img_filtered3))

    img_filtered4 = img_ft_centered * kernel4
    img_filtered4_ifft = np.abs(np.fft.ifft2(img_filtered4))

    # Save images
    img_set1 = [img_rgb, img_gray, magnitude_spectrum, centered_magnitude_spectrum, kernel1, img_filtered1_ifft]
    img_set2 = [img_rgb, img_gray, magnitude_spectrum, centered_magnitude_spectrum, kernel2, img_filtered2_ifft]
    img_set3 = [img_rgb, img_gray, magnitude_spectrum, centered_magnitude_spectrum, kernel3, img_filtered3_ifft]
    img_set4 = [img_rgb, img_gray, magnitude_spectrum, centered_magnitude_spectrum, kernel4, img_filtered4_ifft]
    img_set = [img_set1, img_set2, img_set3, img_set4]
    # img_kernel = [kernel1, kernel2, kernel3, kernel4]
    # img_filtered = [img_filtered1_ifft, img_filtered2_ifft, img_filtered3_ifft, img_filtered4_ifft]
    img_title = ['RGB', 'Gray', 'FFT2', 'Centered FFT2', 'Filter', 'Filtered Imgae']

    for i in range(4):
        img_plot(img_set[i], img_title, i+1)

def img_plot(img_set, img_title, cnt):
    plt.figure(figsize = (20, 20))
    for i in range(len(img_set)):
        plt.subplot(2, 3, i+1)
        plt.title(img_title[i])
        if (i == 0):
            plt.imshow(img_set[i])
        else:
            plt.imshow(img_set[i], cmap = 'gray')
    
    plt.savefig('fig-' + str(cnt) + '.jpg')
    plt.show()

if __name__ == '__main__':
    main()