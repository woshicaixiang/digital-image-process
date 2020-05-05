import numpy as np
import cv2 as cv
import addNoise as add
from matplotlib import pyplot as plt
import os


def addSalt_pepperNoise(image, snr, gray=True):
	"""
	:param image: 输入图像
	:param snr: 信噪比
	:param gray: 输入图像是否为灰度图像
	:return: 添加椒盐噪声后的图像
	
	"""
	height, width = image.shape[0], image.shape[1]
	imageNew = image.copy()
	num1 = height * width
	num2 = int(num1 * (1 - snr)) #计算需要添加的噪声点个数
	for i in range(num2):
		# 随机产生噪声点坐标
		randX = np.random.randint(1, height - 1)
		randY = np.random.randint(1, width - 1)
		# 添加椒盐噪声（此处仅添加灰度值为0和255的噪声）
		if gray:
			if np.random.random() <= 0.5:
				imageNew[randX, randY] = 0
			else:
				imageNew[randX, randY] = 255
		else:
			if np.random.random() <= 0.5:
				imageNew[randX, randY, 0:3] = 0
			else:
				imageNew[randX, randY, 0:3] = 255
	return imageNew


def medianFilter(image, size, gray=True):
	"""
	:param image: 输入图像
	:param size: 滤波器尺寸
	:param gray: 输入图像是否为灰度图像
	:return: 返回中值滤波后的图像
	
	"""
	height, width = image.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE) #对图像进行扩展
	imageNew = image.copy()
	if gray:
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = np.median(imageExband[i:i+size, j:j+size])
	else:
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = np.median(imageExband[i:i+size, j:j+size, ch])
	return imageNew


def averageFilter(image, size, gray=True):
	"""
	:param image: 输入图像
	:param size: 滤波器尺寸
	:param gray: 输入图像是否为灰度图像
	:return: 返回均值滤波后的图像
	
	"""
	height, width = image.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE)#对图像进行扩展
	imageNew = image.copy()
	if gray:
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = np.mean(imageExband[i:i+size, j:j+size])
	else:
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = np.mean(imageExband[i:i+size, j:j+size, ch])
	return imageNew


def adapt_medianFilter(image, size, sizeMax, gray=True):
	"""
	
	:param image: 输入图像
	:param size: 滤波器初始尺寸
	:param sizeMax: 滤波器最大尺寸
	:param gray: 输入图像是否为灰度图像
	:return: 返回自适应中值滤波后的图像
	
	"""
	height,width = image.shape[0], image.shape[1]
	exband = np.uint((sizeMax - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE)#对图像进行扩展
	imageNew = image.copy()
	if gray:
		for i in range(height):
			for j in range(width):
				A1 = np.int16(np.median(imageExband[i:i+size,j:j+size])) - np.int16(
					np.min(imageExband[i:i+size,j:j+size]))
				A2 = np.int16(np.median(imageExband[i:i+size,j:j+size])) - np.int16(
					np.max(imageExband[i:i+size,j:j+size]))
				B1 = np.int16(image[i, j]) - np.int16(np.min(imageExband[i:i+size,j:j+size]))
				B2 = np.int16(image[i, j]) - np.int16(np.max(imageExband[i:i+size,j:j+size]))
				# A层
				if  A1 > 0 and A2 < 0:
					# B层
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
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					A1 = np.int16(np.median(imageExband[i:i + size, j:j + size, ch])) - np.int16(
						np.min(imageExband[i:i + size, j:j + size, ch]))
					A2 = np.int16(np.median(imageExband[i:i + size, j:j + size, ch])) - np.int16(
						np.max(imageExband[i:i + size, j:j + size, ch]))
					B1 = np.int16(image[i, j, ch]) - np.int16(np.min(imageExband[i:i + size, j:j + size, ch]))
					B2 = np.int16(image[i, j, ch]) - np.int16(np.max(imageExband[i:i + size, j:j + size, ch]))
					# A层
					if A1 > 0 and A2 < 0:
						# B层
						if B1 > 0 and B2 < 0:
							imageNew[i, j, ch] = image[i, j, ch]
						else:
							imageNew[i, j, ch] = np.median(imageExband[i:i + size, j:j + size, ch])
					else:
						if size > sizeMax:
							imageNew[i, j, ch] = image[i, j, ch]
						else:
							size = size + 2
	return imageNew


def adapt_averageFilter(image, size, sigma_2, gray=True):
	"""
	
	:param image: 输入图像
	:param size: 滤波器尺寸
	:param sigma_2: 噪声方差估计值
	:param gray: 输入图像是否为灰度图像
	:return: 返回自适应均值滤波后的图像
	"""

	imageNew = image.copy()
	height, width = image.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(image, exband, exband, exband, exband, cv.BORDER_REPLICATE) #对图像进行扩展
	if gray:
		for i in range(height):
			for j in range(width):
				imageNew[i, j] = image[i, j] - (sigma_2/(np.var(imageExband[i:i+size, j:j+size]))) * (
					image[i, j] - np.mean(imageExband[i:i+size, j:j+size]))
	else:
		for ch in range(3):
			for i in range(height):
				for j in range(width):
					imageNew[i, j, ch] = image[i, j, ch] - (sigma_2/(np.var(imageExband[i:i+size, j:j+size, ch]))) * (
						image[i, j, ch] - np.mean(imageExband[i:i+size, j:j+size, ch]))
	return imageNew


if __name__ == '__main__':
	# 新建文件夹
	if os.path.exists(r'./pro2'):
		cur_path = os.getcwd()
		cur_path = cur_path + '\\pro2'
		cur_file = os.listdir('pro2')
		for i in cur_file:
			os.remove(cur_path + str('\\') + i)
	else:
		os.mkdir('pro2')

	# 读取原始图像
	img = cv.imread(r'./lena.jpg')
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# 添加高斯噪声和椒盐噪声
	img1, noise1 = add.gaussian_blur(img_gray, 0, 30)
	img2, noise2 = add.gaussian_blur(img, 0, 30, False)
	img_sp1 = addSalt_pepperNoise(img_gray,0.3)
	img_sp2 = addSalt_pepperNoise(img, 0.3, False)
	# 对图像进行滤波操作
	output1 = medianFilter(img_sp1, 3)
	output2 = adapt_medianFilter(img_sp1, 1, 5)
	output3 = medianFilter(img_sp2, 3, False)
	output4 = adapt_medianFilter(img_sp2, 1, 5, False)

	output5 = averageFilter(img1, 3)
	output6 = adapt_averageFilter(img1, 3, 900)
	output7 = averageFilter(img2, 3, False)
	output8 = adapt_averageFilter(img2, 3, 900, False)

	# 绘制并保存直方图
	plt.figure(1)
	histOutput1 = plt.hist(output1.ravel(), 256)
	plt.savefig(r'./pro2/histOutput1.png')
	plt.figure(2)
	histOutput2 = plt.hist(output2.ravel(), 256)
	plt.savefig(r'./pro2/histOutput2.png')
	plt.figure(3)
	histOutput3 = plt.hist(output3.ravel(), 256)
	plt.savefig(r'./pro2/histOutput3.png')
	plt.figure(4)
	histOutput4 = plt.hist(output4.ravel(), 256)
	plt.savefig(r'./pro2/histOutput4.png')
	plt.figure(5)
	histOutput5 = plt.hist(output5.ravel(), 256)
	plt.savefig(r'./pro2/histOutput5.png')
	plt.figure(6)
	histOutput6 = plt.hist(output6.ravel(), 256)
	plt.savefig(r'./pro2/histOutput6.png')
	plt.figure(7)
	histOutput7 = plt.hist(output7.ravel(), 256)
	plt.savefig(r'./pro2/histOutput7.png')
	plt.figure(8)
	histOutput8 = plt.hist(output8.ravel(), 256)
	plt.savefig(r'./pro2/histOutput8.png')
	plt.figure(10)
	histImg1 = plt.hist(img1.ravel(), 256)
	plt.savefig(r'./pro2/histImg1.png')
	plt.figure(11)
	histImg2 = plt.hist(img2.ravel(), 256)
	plt.savefig(r'./pro2/histImg2.png')
	plt.figure(12)
	histImg_sp1 = plt.hist(img_sp1.ravel(), 256)
	plt.savefig(r'./pro2/histImg_sp1.png')
	plt.figure(13)
	histImg_sp2 = plt.hist(img_sp2.ravel(), 256)
	plt.savefig(r'./pro2/histImg_sp2.png')
	# plt.show()

	# 显示图像并保存
	cv.imshow('01', img_sp1)
	cv.imwrite(r'./pro2/img_sp1.png',img_sp1)
	cv.imshow('02', img_sp2)
	cv.imwrite(r'./pro2/img_sp2.png',img_sp2)
	cv.imshow('03', img1)
	cv.imwrite(r'./pro2/img1.png',img1)
	cv.imshow('04', img2)
	cv.imwrite(r'./pro2/img2.png',img2)
	cv.imshow('1', output1)
	cv.imwrite(r'./pro2/output1.png',output1)
	cv.imshow('2', output2)
	cv.imwrite(r'./pro2/output2.png',output2)
	cv.imshow('3', output3)
	cv.imwrite(r'./pro2/output3.png',output3)
	cv.imshow('4', output4)
	cv.imwrite(r'./pro2/output4.png',output4)
	cv.imshow('5', output5)
	cv.imwrite(r'./pro2/output5.png',output5)
	cv.imshow('6', output6)
	cv.imwrite(r'./pro2/output6.png',output6)
	cv.imshow('7', output7)
	cv.imwrite(r'./pro2/output7.png',output7)
	cv.imshow('8', output8)
	cv.imwrite(r'./pro2/output8.png',output8)




	cv.waitKey()
	cv.destroyAllWindows()
