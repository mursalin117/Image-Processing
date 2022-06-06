import matplotlib.pyplot as plt
import cv2

def main():
	img_path = './field.jpg'
	print(img_path)
	rgb = plt.imread(img_path)
	
	red = rgb[:, :, 0]
	green = rgb[:, :, 1]
	blue = rgb[:, :, 2]

	grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

	_, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)

	img_set = [rgb, red, green, blue, grayscale, binary]
	title_set = ['RGB', 'Red', 'Green', 'Blue', 'Grayscale', 'Binary']
	plt.figure(figsize = (20, 20))
	for i in range(6):
		plt.subplot(2, 3,  i + 1)
		plt.title(title_set[i])
		if (i == 0):
			plt.hist(red.ravel(), 256, [0, 256])
			plt.hist(green.ravel(), 256, [0, 256])
			plt.hist(blue.ravel(), 256, [0, 256])
		else:
			plt.hist(img_set[i].ravel(), 256, [0, 256])
	
	plt.show()

if __name__ == '__main__':
	main()
