import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.models import Model, load_model

IMG_SHAPE = (28,28)

def main():    
    model_path = 'Deblurring_CNN.h5'
    model = load_model(model_path)
    
    path = './testdata/'
    testX,testY = prepareTestDataset(path)

    output = model.predict(testX)

    img_set = [testX[:3, :, :, 0], output[:3, :, :, 0], testY[:3, :, :, 0]]
    title_set = ['Blurred Image', 'Deblurred Image', 'Clear Image']
    plot_img(img_set,title_set)


def generate_blurred_img(img_set):
	n = img_set.shape[0]
	blurred_img_set = img_set.copy()
	for i in range(n):
		blurred_img_set[i] = cv2.medianBlur(img_set[i], 5)

	return blurred_img_set

def prepareTestDataset(path):
    imgs = []
    for file in os.listdir(path):
        img = plt.imread(path+file)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img,IMG_SHAPE)
        imgs.append(img)
    
    testY = np.array(imgs)
    testX = generate_blurred_img(testY)
    testY = np.expand_dims(testY,axis=3)
    testX = np.expand_dims(testX,axis=3)
    print("testX shape: ",testX.shape)
    return testX,testY

def plot_img(img_set, title_set):
	plt.figure(figsize = (20, 20))
	# plt.rcParams['font.size'] = 4
	n = len(img_set)
	k = 1
	for i in range(n):
		for j in range(3):
			plt.subplot(n, 3, k)
			plt.imshow(img_set[i][j], cmap = 'gray')
			plt.axis('off')
			plt.title(title_set[i])
			k += 1

	plt.savefig('fig-1.jpg')
	plt.show()

if __name__ == '__main__':
    main()