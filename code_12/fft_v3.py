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
    row, col = img_gray.shape
    mask = np.ones((row, col), dtype=np.uint8)
    center = [int(row/2), int(col/2)]
    r = 80
    x, y = np.ogrid[:row, :col]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    # gaussian low pass filter
    mask2 = gaussianLPF(50, img_gray.shape[:2]) 
    # print(mask2)
    # gaussian filter high pass = 1 - gaussianLPF(50, img_gray.shape[:2])

    # Butterworth low pass filter
    mask3 = butterworthLPF(100, 2, img_gray.shape[:2])
    # print(mask3)
    # butterworth filter high pass = 1 - butterworthLPF(100, 2, img_gray.shape[:2])
    
    # Apply the filters
    img_filtered1 = img_ft_centered * mask
    img_filtered1_ishift = np.fft.ifftshift(img_filtered1)
    img_filtered1_ishift_ifft = np.abs(np.fft.ifft2(img_filtered1_ishift))

    img_filtered2 = img_ft_centered * mask2
    img_filtered2_ishift = np.fft.ifftshift(img_filtered2)
    img_filtered2_ishift_ifft = np.abs(np.fft.ifft2(img_filtered2_ishift))

    img_filtered3 = img_ft_centered * mask3
    img_filtered3_ishift = np.fft.ifftshift(img_filtered3)
    img_filtered3_ishift_ifft = np.abs(np.fft.ifft2(img_filtered3_ishift))


    # Save images
    img_set1 = [img_rgb, img_gray, magnitude_spectrum, centered_magnitude_spectrum, mask, img_filtered1_ishift_ifft]
    img_set2 = [img_rgb, img_gray, magnitude_spectrum, centered_magnitude_spectrum, mask2, img_filtered2_ishift_ifft]
    img_set3 = [img_rgb, img_gray, magnitude_spectrum, centered_magnitude_spectrum, mask3, img_filtered3_ishift_ifft]
    img_set = [img_set1, img_set2, img_set3]
    img_title = ['RGB', 'Gray', 'FFT2', 'Centered FFT2', 'Filter', 'Filtered Imgae']

    for i in range(len(img_set)):
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
    
    plt.savefig('fig-v3' + str(cnt) + '.jpg')
    plt.show()

def distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

# Gaussian lowpass filter
def gaussianLPF(D0, img_shape):
    row, col = img_shape
    mask = np.zeros((row, col))
    center = [int(row/2), int(col/2)]
    for i in range(row):
        for j in range(col):
            mask[i, j] = np.exp((-distance((i, j), center)**2)/(2*(D0**2)))
    
    return mask

# Butterworth lowpass filter
def butterworthLPF(D0, n, img_shape):
    row, col = img_shape
    mask = np.zeros((row, col))
    center = [int(row/2), int(col/2)]
    for i in range(row):
        for j in range(col):
            mask[i, j] = 1/(1+((distance((i, j), center)/D0))**(2*n))
    
    return mask


if __name__ == '__main__':
    main()