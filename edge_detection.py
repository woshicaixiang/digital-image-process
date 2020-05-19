import numpy as np
import cv2 as cv
import math


def getGaussianCore(size=3, sigma=1):
	"""
	"""
	gaussianCore = np.zeros((size,size))
	d = np.uint((size + 1) / 2)
	for i in range(size):
		for j in range(size):
			gaussianCore[i,j] = \
				math.exp(- ((i-d)**2 + (j-d)**2) / (2 * (sigma ** 2))) / (((2 * math.pi) ** 0.5) * (sigma ** 2))
	return gaussianCore


def gaussianFilter(image, size=3, sigma=1.5, gray=True):
	imageNew = image.copy()
	height, width = image.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE)
	gCore = getGaussianCore(size, sigma)
	for i in range(height):
		for j in range(width):
			imageNew[i, j] = np.sum(gCore * imageExband[i:i+size,j:j+size])
	return imageNew


def sobelMethod(image):
	height, width = image.shape[0], image.shape[1]
	imageExband = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
	gx = np.zeros((height, width))
	gy = np.zeros((height, width))
	gradient = np.zeros((height, width))
	alpha = np.zeros((height, width))
	model = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
					  [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
					  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
					  [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]])
	for i in range(height):
		for j in range(width):
			gx[i, j] = np.sum(model[0,:,:] * imageExband[i:i+3,j:j+3])
			gy[i, j] = np.sum(model[1, :, :] * imageExband[i:i + 3, j:j + 3])
			gradient[i ,j] = np.sqrt(np.square(gx[i, j]) + np.square(gy[i, j]))
			alpha[i, j] = np.arctan(gx[i, j] / gy[i, j])
	return gradient, alpha


def getGradient(image):
	height, width = image.shape[0], image.shape[1]
	dx = np.zeros((height, width))
	dy = np.zeros((height, width))
	gradient = np.zeros((height, width))
	alpha = np.zeros((height, width))
	for i in range(height - 1):
		for j in range(width - 1):
			dx[i, j] = image[i, j + 1] - image[i, j]
			dy[i ,j] = image[i + 1, j] - image[i, j]
			gradient[i ,j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
			alpha[i, j] = np.arctan(dy[i, j] / dx[i, j])
	return gradient, alpha


def nmsMethod(image):
	imageNew = image.copy()
	height, width = image.shape[0], image.shape[1]
	gra, arg = sobelMethod(image)
	# gra, arg = getGradient(image)
	gra = cv.copyMakeBorder(gra, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
	direction = np.array([0, np.pi/4, np.pi/2, -np.pi/4, -np.pi/2])
	diff = np.zeros((1, 5))
	for i in range(height - 1):
		for j in range(width - 1):
			for k in range(5):
				diff[0, k] = abs(arg[i, j] - direction[k])
			min_index = np.argmin(diff)
			condition = np.array([
								 gra[i + 1, j + 1] >=  gra[i + 1, j] and gra[i + 1, j + 1] >= gra[i + 1, j + 2],
								 gra[i + 1, j + 1] >= gra[i, j] and gra[i + 1, j + 1] >= gra[i + 2, j + 2],
								 gra[i + 1, j + 1] >= gra[i, j + 1] and gra[i + 1, j + 1] >= gra[i + 2, j + 1],
								 gra[i + 1, j + 1] >= gra[i + 2, j] and gra[i + 1, j + 1] >= gra[i, j + 2]
								 ])
			if min_index == 2 or min_index == 4:
				if condition[2]:
					imageNew[i, j] = gra[i ,j]
				else:
					imageNew[i, j] = 0
			else:
				if condition[min_index]:
					imageNew[i, j] = gra[i ,j]
				else:
					imageNew[i, j] = 0
			# if min_index == 0:
			# 	if condition[2]:
			# 		imageNew[i, j] = image[i ,j]
			# 	else:
			# 		imageNew[i, j] = 0
			# elif min_index == 1:
			# 	if condition[3]:
			# 		imageNew[i, j] = image[i ,j]
			# 	else:
			# 		imageNew[i, j] = 0
			# elif min_index == 2 or min_index == 4:
			# 	if condition[0]:
			# 		imageNew[i, j] = image[i ,j]
			# 	else:
			# 		imageNew[i, j] = 0
			# elif min_index == 3:
			# 	if condition[1]:
			# 		imageNew[i, j] = image[i ,j]
			# 	else:
			# 		imageNew[i, j] = 0
	return imageNew

def exJudge(windows, TH):
	windows = windows.reshape(9)
	for i in range(9):
		if windows[i] > TH:
			return True
	return False


def thresholdProcess(image, TH, TL):
	imageNew = image.copy()
	height, width = image.shape[0], image.shape[1]
	# imageExband = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
	imageExband = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_CONSTANT,value=0)
	for i in range(height):
		for j in range(width):
			if imageExband[i + 1, j + 1] > TH:
				imageNew[i, j] = 255
			elif imageExband[i + 1, j + 1] < TL:
				imageNew[i, j] = 0
			elif exJudge(imageExband[i:i+3, j:j+3], TH):
				imageNew[i, j] = 255
			else:
				imageNew[i, j] = 0
	return  imageNew

def cannyMethod(image, TH, TL, gray=True):
	if not gray:
		imageNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	else:
		imageNew = image.copy()

	imageNew = gaussianFilter(imageNew)
	# cv.imshow('1',imageNew)
	# imageNew, arg = getGradient(imageNew)

	# imageNew, arg = sobelMethod(imageNew)
	# cv.imshow('2', imageNew)
	imageNew = nmsMethod(imageNew)
	# cv.imshow('3',imageNew)
	imageNew = thresholdProcess(imageNew, TH, TL)
	# cv.imshow('4',imageNew)
	# imageNew = cv.Canny(image,50,120)
	return imageNew


# def connect(image, Tm, Ta):


if __name__ == '__main__':
	image = cv.imread(r'./lena.jpg')
	# image = cv.resize(image, (int(image.shape[1] / 3), int(image.shape[0] / 3)),interpolation=cv.INTER_CUBIC)
	canny1 = cannyMethod(image, 0.2*np.max(image), 0.1*np.max(image), False)
	canny2 = cv.Canny(image, 0.2*np.max(image), 0.6*np.max(image))
	# img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	cv.imshow('origin', image)
	cv.imshow('canny1', canny1)
	cv.imshow('canny2', canny2)

	cv.waitKey()
	cv.destroyAllWindows()