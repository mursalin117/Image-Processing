# solution

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def main():
    img_path = './trees_in_water.jpg'
    print(img_path)

    img_rgb = plt.imread(img_path)
    print(img_rgb.shape)

    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    print(img_gray.shape)
    cnt = 0
    # plt.imshow(img_gray, cmap='gray')
    # plt.savefig('fig-0.jpg')
    # plt.show()
    plt.imsave('./fig-' + str(cnt) + '.jpg', img_gray, format='jpg', cmap='gray')

    img_set = [img_gray]
    img_title = ['Grayscale']

    ft = np.fft.fft2(img_gray)
    ftSort = np.sort(np.abs(ft.reshape(-1))) # sort by highest magnitude

    cnt = 0
    # Zero out all the small co-efficients and inverse transformation
    for keep in (0.7, 0.3, 0.1):
        thresh = ftSort[int(np.floor((1-keep)*len(ftSort)))]
        ind = np.abs(ft) > thresh           # find small indices
        Atlow = ft * ind                    # threshold small indices
        ftlow = np.fft.ifft2(Atlow).real    # compressed image
        img_set.append(ftlow)
        img_title.append('Image at ' + str(keep*100) + '%' + ' compression')
        cnt += 1
        plt.imsave('./fig-' + str(cnt) + '.jpg', ftlow, format='jpg', cmap='gray')
        print(ftlow.shape)

    img_plot(img_set, img_title)

def img_plot(img_set, img_title):
    n = len(img_set)
    plt.figure(figsize = (20, 20))
    for i in range(n):
        plt.subplot(2, 4, i+1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(img_title[i])
        
    plt.savefig('output.jpg')
    plt.show()

if __name__ == '__main__':
    main()

# image save option
# import matplotlib.image as mpimg

# img = mpimg.imread("src.png")
# mpimg.imsave("out.png", img)


# previous code

# import matplotlib.pyplot as plt
# import numpy as np
# import cv2 as cv

# def main():
#     img_path = './cat.jpg'
#     print(img_path)

#     img_gray = cv.imread(img_path, 0)
#     print(img_gray.shape)

#     ft = np.fft.fft2(img_gray)
#     ftSort = np.sort(np.abs(ft.reshape(-1))) 

#     for keep in (0.7, 0.3, 0.1):
#         thresh = int(np.floor((1-keep)*len(ftSort)))
#         ind = ftSort > thresh
#         Atlow = ftSort * ind
#         ftlow = np.fft.ifft2(Atlow).real
#         plt.imshow(ftlow)
#         plt.show()

# if __name__ == '__main__':
#     main()