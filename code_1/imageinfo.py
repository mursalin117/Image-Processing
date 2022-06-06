import matplotlib.pyplot as plt
import cv2

def main():
	img_path = './sunset.jpg'
	print(img_path)

	rgb_img = plt.imread(img_path)
	# img = cv2.imread(img_path)	
	
	print(rgb_img.shape)
	print(rgb_img)
	
	grayscale_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
	print(grayscale_img.shape)
	
	_,binary_img = cv2.threshold(grayscale_img, 127, 255, cv2.THRESH_BINARY)
	#binary_img = cv2.cvtColor(grayscale_img, 127, 255, cv2.THRESH_BINARY)
	print(binary_img.shape)
	
	plt.figure(figsize = (20, 20))
	
	plt.subplot(2, 3, 1)
	plt.title('RGB')
	plt.imshow(rgb_img)
	
	plt.subplot(2, 3, 2)
	plt.title('Red')
	plt.imshow(rgb_img[:, :, 0], cmap = 'gray')
	
	plt.subplot(2, 3, 3)
	plt.title('Green')
	plt.imshow(rgb_img[:, :, 1], cmap = 'gray')
	
	plt.subplot(2, 3, 4)
	plt.title('Blue')
	plt.imshow(rgb_img[:, :, 2], cmap = 'gray')
	
	plt.subplot(2, 3, 5)
	plt.title('Grayscale')
	plt.imshow(grayscale_img, cmap = 'gray')
	
	plt.subplot(2, 3, 6)
	plt.title('Binary')
	plt.imshow(binary_img, cmap = 'gray')
	
	plt.show()
	
if __name__ == '__main__':
	main()
