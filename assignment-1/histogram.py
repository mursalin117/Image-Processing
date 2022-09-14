import matplotlib.pyplot as plt
import cv2 as cv

def main():
	img_path = './trees_in_water_3.jpg'
	print(img_path)
	
	rgb = plt.imread(img_path)

	red = rgb[:, :, 0]
	green = rgb[:, :, 1]
	blue = rgb[:, :, 2]

	histRed = cv.calcHist([rgb],[0],None,[256],[0,256])
	histGreen = cv.calcHist([rgb],[1],None,[256],[0,256])
	histBlue = cv.calcHist([rgb],[2],None,[256],[0,256])

	grayscale = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
	histGrayscale = cv.calcHist([grayscale],[0],None,[256],[0,256])

	_, binary = cv.threshold(grayscale, 127, 255, cv.THRESH_BINARY)
	histBinary = cv.calcHist([binary],[0],None,[256],[0,256])

	img_set = [rgb, red, green, blue, grayscale, binary]
	hist_set = [rgb, histRed, histGreen, histBlue, histGrayscale, histBinary]
	title_set = ['RGB', 'Red', 'Green', 'Blue', 'Grayscale', 'Binary']
	
	plt.figure(figsize = (20, 20))
	for i in range(6):
		plt.subplot(2, 3,  i + 1)
		plt.title(title_set[i])
		ch = len(img_set[i].shape)
		if (ch == 3):
			plt.imshow(img_set[i])
		elif (i == 1 or i == 2 or i == 3):
			plt.imshow(img_set[i], cmap = title_set[i] + 's')
		else: 
			plt.imshow(img_set[i], cmap = 'gray')			
	plt.savefig('fig-1.jpg')
	plt.show()
	
	plt.figure(figsize = (15, 15))
	for i in range(6):
		plt.subplot(2, 3,  i + 1)
		plt.title(title_set[i])
		if (i == 0):
			plt.plot(histRed, 'r')
			plt.plot(histGreen, 'g')
			plt.plot(histBlue, 'b')
		else:
			plt.plot(hist_set[i])
	plt.savefig('fig-2.jpg')
	plt.show()
	

if __name__ == '__main__':
	main()
