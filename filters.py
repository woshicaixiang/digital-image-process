import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


def getMedian(block):
	median = np.median(block)
	return  median


def getMean(block):
	mean = np.mean(block)
	return  mean


def getMax(block):
	max = np.max(block)
	return  max

def getMin(block):
	min = np.min(block)
	return  min


def medianFilter(image, size, gray=True):
	height, width = image.shape[0], image.shape[1]
	if gray:
		imageNew = np.zeros((height, width),dtype=np.uint8)
		imageExband = np.zeros((height+2, width+2),dtype=np.uint8)
		imageExband[1:height+1, 1:width+1] = image[0:height, 0:width]
		imageExband[0, 1:width+1] = image[0, 0:width]
		imageExband[height+1, 1:width + 1] = image[height-1, 0:width]
		imageExband[1:height+1, 0] = image[0:height, 0]
		imageExband[1:height + 1, width+1] = image[0:height, width-1]
		# print(imageExband)
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = getMedian(imageExband[i:i+3, j:j+3])
	# imageaaa = np.zeros((height, width))
	# imageaaa = imageNew[0:height,0:width,0]
	else:
		imageNew = np.zeros((height, width, 3), dtype=np.uint8)
		imageExband = np.zeros((height + 2, width + 2, 3), dtype=np.uint8)
		imageExband[1:height + 1, 1:width + 1, 0:3] = image[0:height, 0:width, 0:3]
		imageExband[0, 1:width + 1, 0:3] = image[0, 0:width, 0:3]
		imageExband[height + 1, 1:width + 1, 0:3] = image[height - 1, 0:width, 0:3]
		imageExband[1:height + 1, 0, 0:3] = image[0:height, 0, 0:3]
		imageExband[1:height + 1, width + 1, 0:3] = image[0:height, width - 1, 0:3]
		# print(imageExband)
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = getMedian(imageExband[i:i + 3, j:j + 3, ch])

	# print(imageNew)
	return imageNew


def averageFilter(image, size, gray=True):
	height, width = image.shape[0], image.shape[1]
	if gray:
		imageNew = np.zeros((height, width),dtype=np.uint8)
		imageExband = np.zeros((height+2, width+2),dtype=np.uint8)
		imageExband[1:height+1, 1:width+1] = image[0:height, 0:width]
		imageExband[0, 1:width+1] = image[0, 0:width]
		imageExband[height+1, 1:width + 1] = image[height-1, 0:width]
		imageExband[1:height+1, 0] = image[0:height, 0]
		imageExband[1:height + 1, width+1] = image[0:height, width-1]
		# print(imageExband)
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = getMean(imageExband[i:i+3, j:j+3])
	# imageaaa = np.zeros((height, width))
	# imageaaa = imageNew[0:height,0:width,0]
	else:
		imageNew = np.zeros((height, width, 3), dtype=np.uint8)
		imageExband = np.zeros((height + 2, width + 2, 3), dtype=np.uint8)
		imageExband[1:height + 1, 1:width + 1, 0:3] = image[0:height, 0:width, 0:3]
		imageExband[0, 1:width + 1, 0:3] = image[0, 0:width, 0:3]
		imageExband[height + 1, 1:width + 1, 0:3] = image[height - 1, 0:width, 0:3]
		imageExband[1:height + 1, 0, 0:3] = image[0:height, 0, 0:3]
		imageExband[1:height + 1, width + 1, 0:3] = image[0:height, width - 1, 0:3]
		# print(imageExband)
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = getMean(imageExband[i:i + 3, j:j + 3, ch])

	# print(imageNew)
	return imageNew

# def adapt_averageFilter(image, sizeOrigin, sizeMax, gray=True):



if __name__ == '__main__':
	img_color = cv.imread(r'./gaussian_blur/img_color1g.png')
	img_gray = cv.imread(r'./gaussian_blur/img_gray1g.png')

	# print(img_color.shape[2])
	img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)
	# print(img_gray)
	# print(img_gray.shape)
	p1 = medianFilter(img_gray)
	p2 = medianFilter(img_color,False)
	# print(p1)
	# print(p1.shape)
	# p1 = cv.cvtColor(p1, cv.COLOR_BGR2GRAY)
	cv.imshow('1', p1)
	cv.imshow('2', p2)
	cv.imshow('3', img_color)
	# cv.imshow('2', img_gray)

	cv.waitKey()
	cv.destroyAllWindows()
