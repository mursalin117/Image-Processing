import matplotlib.pyplot as plt
import cv2

img_path = 'girls.jpg'

plt.subplot(2,3,1)
plt.title('RGB')
rgb = plt.imread(img_path)
plt.imshow(rgb)
plt.show()

plt.subplot(2,3,2)
plt.title('RED')
red = rgb[:,:,0]
plt.imshow(red, cmap = 'gray')
plt.show()

plt.subplot(2,3,3)
plt.title('GREEN')
green = rgb[:,:,1]
plt.imshow(green, cmap = 'gray')
plt.show()

plt.subplot(2,3,4)
plt.title('BLUE')
blue = rgb[:,:,2]
plt.imshow(blue, cmap = 'gray')
plt.show()

grayscale = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
plt.subplot(2,3,5)
plt.title('gray')
plt.imshow(grayscale, cmap = 'gray')
plt.show()

_, binary = cv2.threshold(grayscale, 50, 255, cv2.THRESH_BINARY)
plt.subplot(2,3,6)
plt.title('binary')
plt.imshow(binary, cmap = 'gray')
plt.show()



