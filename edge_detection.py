import numpy as np
import cv2 as cv
import math
import os


def getGaussianCore(size=7, sigma=1):
	"""
	功能：获得指定尺寸的高斯核（高斯模板）
	:param size: 高斯核（高斯模板）尺寸，默认值为7
	:param sigma: 生成高斯核（高斯模板）的标准差，默认值为1
	:return: 返回指定尺寸的高斯核（高斯模板）
	"""
	gaussianCore = np.zeros((size,size))
	d = np.uint((size - 1) / 2)
	for i in range(size):
		for j in range(size):
			gaussianCore[i, j] = \
				math.exp(- ((i - d) ** 2 + (j - d) ** 2) / (2 * (sigma ** 2))) / ((2 * math.pi) * (sigma ** 2))
	return gaussianCore


def gaussianFilter(image, size=7, sigma=1, gray=True):
	"""
	功能：对图像进行高斯滤波
	:param image: 输入图像
	:param size: 高斯核（高斯模板）的尺寸，默认值为7
	:param sigma: 生成高斯核（高斯模板）的标准差，默认值为1
	:param gray: 标记输入图片是否为灰度图像
	:return: 高斯滤波后的图像
	"""
	if not gray:
		imageNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	else:
		imageNew = image.copy()
	height, width = imageNew.shape[0], image.shape[1]
	exband = np.uint((size - 1) / 2)
	imageExband = cv.copyMakeBorder(imageNew, exband, exband, exband, exband, cv.BORDER_REPLICATE)
	gCore = getGaussianCore(size, sigma)
	for i in range(height):
		for j in range(width):
			imageNew[i, j] = np.sum(gCore * imageExband[i:i+size,j:j+size])
	return imageNew


def sobelMethod(image, gray=True):
	"""
	功能：使用Sobel算子计算图像的梯度幅值与角度
	:param image: 输入图像
	:param gray: 标记输入图片是否为灰度图像
	:return: 图像的梯度幅值与角度
	"""
	if not gray:
		imageNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	else:
		imageNew = image.copy()
	height, width = imageNew.shape[0], imageNew.shape[1]
	imageExband = cv.copyMakeBorder(imageNew, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
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
			gy[i, j] = np.sum(model[0,:,:] * imageExband[i:i+3,j:j+3])
			gx[i, j] = np.sum(model[1, :, :] * imageExband[i:i + 3, j:j + 3])
			gradient[i ,j] = np.sqrt(np.square(gx[i, j]) + np.square(gy[i, j]))
			# alpha[i, j] = np.arctan(gx[i, j] / gy[i, j])
			alpha[i, j] = np.arctan(gy[i, j] / gx[i, j])
	return gradient, alpha


# def getGradient(image, gray=True):
# 	"""
# 	由于效果较差，因此未使用该方法
# 	功能：计算图像的梯度幅值与角度
# 	:param image: 输入图像
# 	:param gray: 标记输入图片是否为灰度图像
# 	:return: 图像的梯度幅值与角度
# 	"""
# 	if not gray:
# 		imageNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# 	else:
# 		imageNew = image.copy()
# 	height, width = imageNew.shape[0], imageNew.shape[1]
# 	dx = np.zeros((height, width))
# 	dy = np.zeros((height, width))
# 	gradient = np.zeros((height, width))
# 	alpha = np.zeros((height, width))
# 	for i in range(height - 1):
# 		for j in range(width - 1):
# 			dx[i, j] = imageNew[i, j + 1] - imageNew[i, j]
# 			dy[i ,j] = imageNew[i + 1, j] - imageNew[i, j]
# 			gradient[i ,j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
# 			alpha[i, j] = np.arctan(dy[i, j] / dx[i, j])
# 	cv.imshow('2', gradient)
# 	cv.imshow('3', alpha)
# 	return gradient, alpha


def nmsMethod(image, gray=True):
	"""
	功能：对图像进行非极大值抑制的操作
	:param image: 输入图像
	:param gray: 标记输入图片是否为灰度图像
	:return: 非极大值抑制后的图像
	"""
	if not gray:
		imageNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	else:
		imageNew = image.copy()
	height, width = imageNew.shape[0], imageNew.shape[1]
	gra, arg = sobelMethod(image)
	# gra, arg = getGradient(image)
	gra = cv.copyMakeBorder(gra, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
	direction = np.array([0, np.pi/4, np.pi/2, -np.pi/4, -np.pi/2])
	diff = np.zeros((1, 5))
	for i in range(height):
		for j in range(width):
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
					imageNew[i, j] = gra[i + 1 ,j + 1]
				else:
					imageNew[i, j] = 0
			else:
				if condition[min_index]:
					imageNew[i, j] = gra[i +1,j + 1]
				else:
					imageNew[i, j] = 0
	return imageNew


def exJudge(windows, TH):
	"""
	功能：判断当前像素点的8领域内是否有强边缘
	:param windows: 当前像素点的8邻域
	:param TH: 较高的阈值
	:return: 当前像素点的8领域内有无强边缘（True,False）
	"""
	windows = windows.reshape(9)
	for i in range(9):
		if windows[i] > TH:
			return True
	return False


def thresholdProcess(image, TH, TL, gray=True):
	"""
	功能：双阈值处理及连接分析
	:param image: 输入图像
	:param TH: 较大的阈值
	:param TL: 较小的阈值
	:return: 双阈值处理及连接分析后的图像
	"""
	if not gray:
		imageNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	else:
		imageNew = image.copy()
	height, width = imageNew.shape[0], imageNew.shape[1]
	# imageExband = cv.copyMakeBorder(image, 1, 1, 1, 1, cv.BORDER_REPLICATE)
	imageExband = cv.copyMakeBorder(imageNew, 1, 1, 1, 1, cv.BORDER_CONSTANT,value=0)
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
	"""
	功能：Canny边缘检测
	:param image: 输入图像
	:param TH: 较大的阈值
	:param TL: 较小的阈值
	:param gray: 标记输入图片是否为灰度图像
	:return: 返回Canny边缘检测后的图像
	"""
	if not gray:
		imageNew = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	else:
		imageNew = image.copy()
	imageNew = gaussianFilter(imageNew)
	imageNew = nmsMethod(imageNew)
	imageNew = thresholdProcess(imageNew, TH, TL)
	return imageNew

if __name__ == '__main__':
	image1 = cv.imread(r'./lena.jpg')
	image2 = cv.imread(r'./caixiang.jpg')
	image2 = cv.resize(image2, (int(image2.shape[1] / 3), int(image2.shape[0] / 3)), interpolation=cv.INTER_CUBIC)

	image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
	image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

	canny1 = cannyMethod(image1_gray, 0.2*np.max(image1_gray), 0.1*np.max(image1_gray))
	canny11 = cv.Canny(image1_gray, 0.2*np.max(image1_gray), 0.6*np.max(image1_gray))

	cv.imshow('origin1', image1_gray)
	cv.imshow('canny1', canny1)
	cv.imshow('canny11', canny11)

	canny2 = cannyMethod(image2_gray, 0.15*np.max(image2_gray), 0.08*np.max(image2_gray))
	canny22 = cv.Canny(image2_gray, 0.2*np.max(image2_gray), 0.6*np.max(image2_gray))

	cv.imshow('origin2', image2_gray)
	cv.imshow('canny2', canny2)
	cv.imshow('canny22', canny22)

	# 生成并显示边缘检测每一步的结果
	img1 = gaussianFilter(image1_gray)
	gra, arg = sobelMethod(img1)
	img2 = nmsMethod(img1)
	img3 = thresholdProcess(img2, 0.2*np.max(image1_gray), 0.1*np.max(image1_gray))
	cv.imshow('img1', img1)
	cv.imshow('gra', gra)
	cv.imshow('arg', arg)
	cv.imshow('img2', img2)
	cv.imshow('img3', img3)


	cv.waitKey()
	cv.destroyAllWindows()

	# 新建文件夹

	if os.path.exists(r'./pro3'):
		cur_path = os.getcwd()
		cur_path = cur_path + '\\pro3'
		cur_file = os.listdir('pro3')
		for i in cur_file:
			os.remove(cur_path + str('\\') + i)
	else:
		os.mkdir('pro3')

	cv.imwrite(r'./pro3/image1_gray.png', image1_gray)
	cv.imwrite(r'./pro3/image2_gray.png', image2_gray)
	cv.imwrite(r'./pro3/canny1.png', canny1)
	cv.imwrite(r'./pro3/canny11.png', canny11)
	cv.imwrite(r'./pro3/canny2.png', canny2)
	cv.imwrite(r'./pro3/canny22.png', canny22)

	cv.imwrite(r'./pro3/img1.png', img1)
	cv.imwrite(r'./pro3/gra.png', gra)
	cv.imwrite(r'./pro3/arg.png', arg)
	cv.imwrite(r'./pro3/img2.png', img2)
	cv.imwrite(r'./pro3/img3.png', img3)