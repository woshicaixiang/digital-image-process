import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


# def getMedian(block):
# 	median = np.median(block)
# 	return  median
#
#
# def getMean(block):
# 	mean = np.mean(block)
# 	return  mean
#
#
# def getMax(block):
# 	max = np.max(block)
# 	return  max
#
# def getMin(block):
# 	min = np.min(block)
# 	return  min


def addSalt_pepperNoise(image,snr):
    h=image.shape[0]
    w=image.shape[1]
    imageNew=image.copy()
    sp = h * w   # 计算图像像素点个数
    NP=int(sp*(1-snr))   # 计算图像椒盐噪声点个数
    for i in range (NP):
        randx=np.random.randint(1,h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy=np.random.randint(1,w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random()<=0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            imageNew[randx,randy]=0
        else:
			imageNew[randx,randy]=255
    return imageNew


def medianFilter(image, size, gray=True):
	height, width = image.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE)
	if gray:
		imageNew = np.zeros((height, width),dtype=np.uint8)
		# print(imageExband)
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = np.median(imageExband[i:i+size, j:j+size])
	# imageaaa = np.zeros((height, width))
	# imageaaa = imageNew[0:height,0:width,0]
	else:
		imageNew = np.zeros((height, width, 3), dtype=np.uint8)
		# print(imageExband)
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = np.median(imageExband[i:i+size, j:j+size, ch])
	# print(imageNew)
	return imageNew


def averageFilter(image, size, gray=True):
	height, width = image.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE)
	if gray:
		imageNew = np.zeros((height, width),dtype=np.uint8)
		# print(imageExband)
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = np.mean(imageExband[i:i+size, j:j+size])
	else:
		imageNew = np.zeros((height, width, 3), dtype=np.uint8)
		# print(imageExband)
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = np.mean(imageExband[i:i+size, j:j+size, ch])
	# print(imageNew)
	return imageNew


def adapt_medianFilter(image, size, sizeMax, gray=True):
	height,width = image.shape[0], image.shape[1]
	exband = np.uint((sizeMax - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE)
	if gray:
		imageNew = np.zeros((height, width), dtype=np.uint8)
		for i in range(height):
			for j in range(width):
				A1 = np.int16(np.median(imageExband[i:i+size,j:j+size])) - np.int16(
					np.min(imageExband[i:i+size,j:j+size]))
				# print(A1)
				A2 = np.int16(np.median(imageExband[i:i+size,j:j+size])) - np.int16(
					np.max(imageExband[i:i+size,j:j+size]))
				# print(A2)
				B1 = np.int16(image[i, j]) - np.int16(np.min(imageExband[i:i+size,j:j+size]))
				B2 = np.int16(image[i, j]) - np.int16(np.max(imageExband[i:i+size,j:j+size]))
				if  A1 > 0 and A2 < 0:
					if B1 > 0 and B2 < 0:
						imageNew[i, j] = image[i, j]
					else:
						imageNew[i, j] = np.median(imageExband[i:i+size,j:j+size])
				else:
					if size > sizeMax:
						imageNew[i, j] = image[i, j]
					else:
						size = size + 2
	else:
		imageNew = np.zeros((height, width, 3), dtype=np.uint8)
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					A1 = np.int16(np.median(imageExband[i:i + size, j:j + size, ch])) - np.int16(
						np.min(imageExband[i:i + size, j:j + size, ch]))
					# print(A1)
					A2 = np.int16(np.median(imageExband[i:i + size, j:j + size, ch])) - np.int16(
						np.max(imageExband[i:i + size, j:j + size, ch]))
					# print(A2)
					B1 = np.int16(image[i, j, ch]) - np.int16(np.min(imageExband[i:i + size, j:j + size, ch]))
					B2 = np.int16(image[i, j, ch]) - np.int16(np.max(imageExband[i:i + size, j:j + size, ch]))
					if A1 > 0 and A2 < 0:
						if B1 > 0 and B2 < 0:
							imageNew[i, j, ch] = image[i, j, ch]
						else:
							imageNew[i, j, ch] = np.median(imageExband[i:i + size, j:j + size, ch])
					else:
						if size > sizeMax:
							imageNew[i, j, ch] = image[i, j, ch]
						else:
							size = size + 2
	# print(imageNew)
	return imageNew


def adapt_averageFilter(image, size, sigma_2, gray=True):
	imageNew = image.copy()
	height, width = image.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE)
	if gray:
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = image[i, j] - (sigma_2/(np.std(imageExband[i:i+size, j:j+size]))) * (
					image[i, j] - np.mean(imageExband[i:i+size, j:j+size]))
	else:
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = image[i, j, ch] - (sigma_2/(np.std(imageExband[i:i+size, j:j+size, ch]))) * (
						image[i, j, ch] - np.mean(imageExband[i:i+size, j:j+size, ch]))
	return imageNew


if __name__ == '__main__':
	# img_color = cv.imread(r'./gaussian_blur/img_color1g.png')
	# img_gray = cv.imread(r'./gaussian_blur/img_gray1g.png')
	img = cv.imread(r'./lena.jpg')
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# print(img_color.shape[2])
	# img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
	img = addSalt_pepperNoise(img,0.6)
	# print(img_gray)
	# print(img_gray.shape)
	p1 = medianFilter(img, 3)
	# p2 = medianFilter(img_color, 3, False)
	# p3 = averageFilter(img_gray, 3)
	p4 = adapt_medianFilter(img,1,5)
	# print(p1)
	# print(img_gray)
	# print(p1.shape)
	# p1 = cv.cvtColor(p1, cv.COLOR_BGR2GRAY)
	cv.imshow('1', p1)
	# cv.imshow('2', p2)
	cv.imshow('3', img)
	# cv.imshow('4', p3)
	# cv.imshow('5', img_gray)
	cv.imshow('6', p4)
	# cv.imshow('2', img_gray)

	cv.waitKey()
	cv.destroyAllWindows()
