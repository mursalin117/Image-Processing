import matplotlib.pyplot as plt
import cv2 as cv

def main():
	img_path = './trees_in_water_2.jpg'
	print(img_path)
	rgb = plt.imread(img_path)

	red = cv.calcHist([rgb],[0],None,[256],[0,256])
	green = cv.calcHist([rgb],[1],None,[256],[0,256])
	blue = cv.calcHist([rgb],[2],None,[256],[0,256])

	gray_img = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
	grayscale = cv.calcHist([gray_img],[0],None,[256],[0,256])

	_, binary_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
	binary = cv.calcHist([binary_img],[0],None,[256],[0,256])

	img_set = [rgb, red, green, blue, grayscale, binary]
	title_set = ['RGB', 'Red', 'Green', 'Blue', 'Grayscale', 'Binary']
	plt.figure(figsize = (15, 15))
	for i in range(6):
		plt.subplot(2, 4,  i + 1)
		plt.title(title_set[i])
		ch = len(img_set[i].shape)
		if (ch == 3):
			plt.imshow(img_set[i])
		else:
			plt.plot(img_set[i])

	plt.subplot(2, 4, 7)
	plt.title('All')
	plt.plot(red, 'r')
	plt.plot(green, 'g')
	plt.plot(blue, 'b')

	plt.show()

if __name__ == '__main__':
	main()
