import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


def uGaussianSin_blur(image, mu=0, sigma=1, gray=True):
	"""
	通过均匀分布随机数产生标准正态分布随机数（方法二Sin），
	再产生特定分布的随机数，得到服从特定正态分布的随机数，
	添加到原始图像上，重新标定，从而达到模糊图像的效果。
	:param image: 传入的图像文件（既可以是彩色图像，也可以是灰度图像）;
	:param mu: Gaussian分布的均值，默认值为0;
	:param sigma: Gaussian分布的标准差，默认值为1;
	:param gray: 标记传入以及返回的图像是否为灰度图像，默认值为True;
	:return: 返回添加Gaussian噪声后的图像;
	
	"""
	height, width = image.shape[0], image.shape[1]
	u1 = np.random.rand(height, width)
	u2 = np.random.rand(height, width)
	if gray:
		noise1 = ((-2 * np.log(u1)) ** 0.5) * np.sin(2 * np.pi * u2)
	else:
		noise1 = np.zeros((height, width, 3))
		for ii in range(3):
			noise1[:, :, ii] = ((-2 * np.log(u1)) ** 0.5) * np.sin(2 * np.pi * u2)
	noise2 = noise1 * sigma + mu
	imageNew = noise2 + image
	imageNew = imageNew.astype(np.uint8)
	# noise2 = noise2.astype(np.uint8)
	return imageNew, noise2


def uGaussianCos_blur(image, mu=0, sigma=1, gray=True):
	"""
	通过均匀分布随机数产生标准正态分布随机数（方法三Cos），
	再产生特定分布的随机数，得到服从特定正态分布的随机数，
	添加到原始图像上，重新标定，从而达到模糊图像的效果。
	:param image: 传入的图像文件（既可以是彩色图像，也可以是灰度图像）;
	:param mu: Gaussian分布的均值，默认值为0;
	:param sigma: Gaussian分布的标准差，默认值为1;
	:param gray: 标记传入以及返回的图像是否为灰度图像，默认值为True;
	:return: 返回添加Gaussian噪声后的图像;
	
	"""
	height, width = image.shape[0], image.shape[1]
	u1 = np.random.rand(height, width)
	u2 = np.random.rand(height, width)
	if gray:
		noise1 = ((-2 * np.log(u1)) ** 0.5) * np.cos(2 * np.pi * u2)
	else:
		noise1 = np.zeros((height, width, 3))
		for ii in range(3):
			noise1[:, :, ii] = ((-2 * np.log(u1)) ** 0.5) * np.cos(2 * np.pi * u2)
	noise2 = noise1 * sigma + mu
	imageNew = noise2 + image
	imageNew = imageNew.astype(np.uint8)
	# noise2 = noise2.astype(np.uint8)
	return imageNew, noise2


def gaussian_blur(image, mu=0, sigma=1, gray=True):
	"""
	:param image: 传入的图像文件（既可以是彩色图像，也可以是灰度图像）;
	:param mu: Gaussian分布的均值，默认值为0;
	:param sigma: Gaussian分布的标准差，默认值为1;
	:param gray: 标记传入以及返回的图像是否为灰度图像，默认值为True;
	:return: 返回添加Gaussian噪声后的图像;
	
	"""
	height, width = image.shape[0], image.shape[1]

	if gray:
		noise1 = np.random.randn(height, width)
	else:
		noise1 = np.random.randn(height, width, 3)
	noise2 = noise1 * sigma + mu
	imageNew = noise2 + image
	imageNew = imageNew.astype(np.uint8)
	# noise2 = noise2.astype(np.uint8)
	return imageNew, noise2


def gamma_blur(image, alfa=1, beta=1, gray=True):
	"""
	:param image: 传入的图像文件（既可以是彩色图像，也可以是灰度图像）;
	:param alfa: gamma分布的参数，被称为形状参数，默认值为1;
	:param beta: gamma分布的参数，被称为逆尺度参数，默认值为1;
	:param gray: 标记传入以及返回的图像是否为灰度图像，默认值为True;
	:return: 返回添加gamma噪声后的图像;
	
	"""
	height, width = image.shape[0], image.shape[1]
	if gray:
		noise1 = np.random.gamma(alfa, beta, (height,width))
	else:
		noise1 = np.random.gamma(alfa, beta, (height,width,3))
	# noise2 = noise1 * sigma + mu
	imageNew = noise1 + image
	imageNew = imageNew.astype(np.uint8)
	# noise1 = noise1.astype(np.uint8)
	return imageNew, noise1


if __name__ == '__main__':
	img = cv.imread(r'./lena.jpg')
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)# 将彩色图像转换为灰度图像

	# 使用guassian_blur()函数进行模糊
	img_gray1g, noise_gray1g = gaussian_blur(gray, 0, 1)
	img_gray2g, noise_gray2g = gaussian_blur(gray, 0, 10)
	img_gray3g, noise_gray3g = gaussian_blur(gray, 0, 50)
	img_gray4g, noise_gray4g = gaussian_blur(gray, 20, 10)
	img_gray5g, noise_gray5g = gaussian_blur(gray, -20, 10)
	a11 = [img_gray1g, img_gray2g, img_gray3g, img_gray4g, img_gray5g]
	img_color1g, noise_color1g = gaussian_blur(img, 0, 1, False)
	img_color2g, noise_color2g = gaussian_blur(img, 0, 10, False)
	img_color3g, noise_color3g = gaussian_blur(img, 0, 50, False)
	img_color4g, noise_color4g = gaussian_blur(img, 20, 10, False)
	img_color5g, noise_color5g = gaussian_blur(img, -20, 10, False)
	a12 = [img_color1g, img_color2g, img_color3g, img_color4g, img_color5g]

	# 使用uGaussianSin_blur()函数进行模糊
	img_gray1s, noise_gray1s = uGaussianSin_blur(gray, 0, 1)
	img_gray2s, noise_gray2s  = uGaussianSin_blur(gray, 0, 10)
	img_gray3s, noise_gray3s  = uGaussianSin_blur(gray, 0, 50)
	img_gray4s, noise_gray4s  = uGaussianSin_blur(gray, 20, 10)
	img_gray5s, noise_gray5s  = uGaussianSin_blur(gray, -20, 10)
	a21 = [img_gray1s, img_gray2s, img_gray3s, img_gray4s, img_gray5s]

	img_color1s, noise_color1s = uGaussianSin_blur(img, 0, 1, False)
	img_color2s, noise_color2s = uGaussianSin_blur(img, 0, 10, False)
	img_color3s, noise_color3s = uGaussianSin_blur(img, 0, 50, False)
	img_color4s, noise_color4s = uGaussianSin_blur(img, 20, 10, False)
	img_color5s, noise_color5s = uGaussianSin_blur(img, -20, 10, False)
	a22 = [img_color1s, img_color2s, img_color3s, img_color4s, img_color5s]

	# 使用uGaussianCos_blur()函数进行模糊
	img_gray1c, noise_gray1c = uGaussianCos_blur(gray, 0, 1)
	img_gray2c, noise_gray2c = uGaussianCos_blur(gray, 0, 10)
	img_gray3c, noise_gray3c = uGaussianCos_blur(gray, 0, 50)
	img_gray4c, noise_gray4c = uGaussianCos_blur(gray, 20, 10)
	img_gray5c, noise_gray5c = uGaussianCos_blur(gray, -20, 10)
	a31 = [img_gray1c, img_gray2c, img_gray3c, img_gray4c, img_gray5c]
	img_color1c, noise_color1c = uGaussianCos_blur(img, 0, 1, False)
	img_color2c, noise_color2c = uGaussianCos_blur(img, 0, 10, False)
	img_color3c, noise_color3c = uGaussianCos_blur(img, 0, 50, False)
	img_color4c, noise_color4c = uGaussianCos_blur(img, 20, 10, False)
	img_color5c, noise_color5c = uGaussianCos_blur(img, -20, 10, False)
	a32 = [img_color1c, img_color2c, img_color3c, img_color4c, img_color5c]

	# 使用gamma()函数进行模糊
	img_gray1m, noise_gray1m = gamma_blur(gray, 1, 10)
	img_gray2m, noise_gray2m = gamma_blur(gray, 1, 20)
	img_gray3m, noise_gray3m = gamma_blur(gray, 10, 10)
	a41 = [img_gray1m, img_gray2m, img_gray3m]
	img_color1m, noise_color1m = gamma_blur(img, 1, 10, False)
	img_color2m, noise_color2m = gamma_blur(img, 1, 20, False)
	img_color3m, noise_color3m = gamma_blur(img, 10, 10, False)
	a42 = [img_color1m, img_color2m, img_color3m]


	# 新建文件夹，用于将图片存到本地
	# 如果已经有同名文件夹，则将文件夹中的文件全部删除
	if os.path.exists(r'./gray_image'):
		cur_path = os.getcwd()
		cur_path = cur_path + '\\gray_image'
		cur_file = os.listdir('gray_image')
		for i in cur_file:
			os.remove(cur_path + str('\\') + i)
	else:
		os.mkdir('gray_image')
	if os.path.exists(r'./gaussian_blur'):
		cur_path = os.getcwd()
		cur_path = cur_path + '\\gaussian_blur'
		cur_file = os.listdir('gaussian_blur')
		for i in cur_file:
			os.remove(cur_path + str('\\') + i)
	else:
		os.mkdir('gaussian_blur')
	if os.path.exists(r'./uGaussianSin_blur'):
		cur_path = os.getcwd()
		cur_path = cur_path + '\\uGaussianSin_blur'
		cur_file = os.listdir('uGaussianSin_blur')
		for i in cur_file:
			os.remove(cur_path + str('\\') + i)
	else:
		os.mkdir('uGaussianSin_blur')
	if os.path.exists(r'./uGaussianCos_blur'):
		cur_path = os.getcwd()
		cur_path = cur_path + '\\uGaussianCos_blur'
		cur_file = os.listdir('uGaussianCos_blur')
		for i in cur_file:
			os.remove(cur_path + str('\\') + i)
	else:
		os.mkdir('uGaussianCos_blur')
	if os.path.exists(r'./gamma'):
		cur_path = os.getcwd()
		cur_path = cur_path + '\\gamma'
		cur_file = os.listdir('gamma')
		for i in cur_file:
			os.remove(cur_path + str('\\') + i)
	else:
		os.mkdir('gamma')

	# 显示图像并保存
	# 原始彩色图像和灰度图像
	cv.imshow('The original image', img)
	cv.imshow('The gray image', gray)

	# guassian_blur()函数进行模糊的图像及其直方图
	cv.imshow('img_gray1g', img_gray1g)
	plt.figure(4)
	noise_grayHist1g = plt.hist(noise_gray1g.ravel(),256)
	plt.savefig(r'./gaussian_blur/noise_grayHist1g.png')
	cv.imshow('img_gray2g', img_gray2g)
	plt.figure(5)
	noise_grayHist2g = plt.hist(noise_gray2g.ravel(),256)
	plt.savefig(r'./gaussian_blur/noise_grayHist2g.png')
	cv.imshow('img_gray3g', img_gray3g)
	plt.figure(6)
	noise_grayHist3g = plt.hist(noise_gray3g.ravel(), 256)
	plt.savefig(r'./gaussian_blur/noise_grayHist3g.png')
	cv.imshow('img_gray4g', img_gray4g)
	plt.figure(7)
	noise_grayHist4g = plt.hist(noise_gray4g.ravel(), 256)
	plt.savefig(r'./gaussian_blur/noise_grayHist4g.png')
	cv.imshow('img_gray5g', img_gray5g)
	plt.figure(8)
	noise_grayHist5g = plt.hist(noise_gray5g.ravel(), 256)
	plt.savefig(r'./gaussian_blur/noise_grayHist5g.png')


	cv.imshow('img_color1g', img_color1g)
	plt.figure(9)
	noise_colorHist1g = plt.hist(noise_color1g.ravel(),256)
	plt.savefig(r'./gaussian_blur/noise_colorHist1g.png')
	cv.imshow('img_color2g', img_color2g)
	plt.figure(10)
	noise_colorHist2g = plt.hist(noise_color2g.ravel(),256)
	plt.savefig(r'./gaussian_blur/noise_colorHist2g.png')
	cv.imshow('img_color3g', img_color3g)
	plt.figure(11)
	noise_colorHist3g = plt.hist(noise_color3g.ravel(),256)
	plt.savefig(r'./gaussian_blur/noise_colorHist3g.png')
	cv.imshow('img_color4g', img_color4g)
	plt.figure(12)
	noise_colorHist4g = plt.hist(noise_color4g.ravel(),256)
	plt.savefig(r'./gaussian_blur/noise_colorHist4g.png')
	cv.imshow('img_color5g', img_color5g)
	plt.figure(13)
	noise_colorHist5g = plt.hist(noise_color5g.ravel(),256)
	plt.savefig(r'./gaussian_blur/noise_colorHist5g.png')


	# uGaussianSin_blur()函数进行模糊的图像及其直方图
	cv.imshow('img_gray1s', img_gray1s)
	plt.figure(14)
	noise_grayHist1s = plt.hist(noise_gray1s.ravel(),256)
	plt.savefig(r'./uGaussianSin_blur/noise_grayHist1s.png')
	cv.imshow('img_gray2s', img_gray2s)
	plt.figure(15)
	noise_grayHist2s = plt.hist(noise_gray2s.ravel(),256)
	plt.savefig(r'./uGaussianSin_blur/noise_grayHist2s.png')
	cv.imshow('img_gray3s', img_gray3s)
	plt.figure(16)
	noise_grayHist3s = plt.hist(noise_gray3s.ravel(), 256)
	plt.savefig(r'./uGaussianSin_blur/noise_grayHist3s.png')
	cv.imshow('img_gray4s', img_gray4s)
	plt.figure(17)
	noise_grayHist4s = plt.hist(noise_gray4s.ravel(), 256)
	plt.savefig(r'./uGaussianSin_blur/noise_grayHist4s.png')
	cv.imshow('img_gray5s', img_gray5s)
	plt.figure(18)
	noise_grayHist5s = plt.hist(noise_gray5s.ravel(), 256)
	plt.savefig(r'./uGaussianSin_blur/noise_grayHist5s.png')


	cv.imshow('img_color1s', img_color1s)
	plt.figure(19)
	noise_colorHist1s = plt.hist(noise_color1s.ravel(),256)
	plt.savefig(r'./uGaussianSin_blur/noise_colorHist1s.png')
	cv.imshow('img_color2s', img_color2s)
	plt.figure(20)
	noise_colorHist2s = plt.hist(noise_color2s.ravel(),256)
	plt.savefig(r'./uGaussianSin_blur/noise_colorHist2s.png')
	cv.imshow('img_color3s', img_color3s)
	plt.figure(21)
	noise_colorHist3s = plt.hist(noise_color3s.ravel(),256)
	plt.savefig(r'./uGaussianSin_blur/noise_colorHist3s.png')
	cv.imshow('img_color4s', img_color4s)
	plt.figure(22)
	noise_colorHist4s = plt.hist(noise_color4s.ravel(),256)
	plt.savefig(r'./uGaussianSin_blur/noise_colorHist4s.png')
	cv.imshow('img_color5s', img_color5s)
	plt.figure(23)
	noise_colorHist5s = plt.hist(noise_color5s.ravel(),256)
	plt.savefig(r'./uGaussianSin_blur/noise_colorHist5s.png')


	# uGaussianCos_blur()函数进行模糊的图像及其直方图

	cv.imshow('img_gray1c', img_gray1c)
	plt.figure(24)
	noise_grayHist1c = plt.hist(noise_gray1c.ravel(),256)
	plt.savefig(r'./uGaussianCos_blur/noise_grayHist1c.png')
	cv.imshow('img_gray2c', img_gray2c)
	plt.figure(25)
	noise_grayHist2c = plt.hist(noise_gray2c.ravel(),256)
	plt.savefig(r'./uGaussianCos_blur/noise_grayHist2c.png')
	cv.imshow('img_gray3c', img_gray3c)
	plt.figure(26)
	noise_grayHist3c = plt.hist(noise_gray3c.ravel(), 256)
	plt.savefig(r'./uGaussianCos_blur/noise_grayHist3c.png')
	cv.imshow('img_gray4c', img_gray4c)
	plt.figure(27)
	noise_grayHist4c = plt.hist(noise_gray4c.ravel(), 256)
	plt.savefig(r'./uGaussianCos_blur/noise_grayHist4c.png')
	cv.imshow('img_gray5c', img_gray5c)
	plt.figure(28)
	noise_grayHist5c = plt.hist(noise_gray5c.ravel(), 256)
	plt.savefig(r'./uGaussianCos_blur/noise_grayHist5c.png')


	cv.imshow('img_color1c', img_color1c)
	plt.figure(29)
	noise_colorHist1c = plt.hist(noise_color1c.ravel(),256)
	plt.savefig(r'./uGaussianCos_blur/noise_colorHist1c.png')
	cv.imshow('img_color2c', img_color2c)
	plt.figure(30)
	noise_colorHist2c = plt.hist(noise_color2c.ravel(),256)
	plt.savefig(r'./uGaussianCos_blur/noise_colorHist2c.png')
	cv.imshow('img_color3c', img_color3c)
	plt.figure(31)
	noise_colorHist3c = plt.hist(noise_color3c.ravel(),256)
	plt.savefig(r'./uGaussianCos_blur/noise_colorHist3c.png')
	cv.imshow('img_color4c', img_color4c)
	plt.figure(32)
	noise_colorHist4c = plt.hist(noise_color4c.ravel(),256)
	plt.savefig(r'./uGaussianCos_blur/noise_colorHist4c.png')
	cv.imshow('img_color5c', img_color5c)
	plt.figure(33)
	noise_colorHist5c = plt.hist(noise_color5c.ravel(),256)
	plt.savefig(r'./uGaussianCos_blur/noise_colorHist5c.png')

	# gamma()函数进行模糊的图像及其直方图
	cv.imshow('img_gray1m', img_gray1m)
	plt.figure(34)
	noise_grayHist1m = plt.hist(noise_gray1m.ravel(),256)
	plt.savefig(r'./gamma/noise_grayHist1m.png')
	cv.imshow('img_gray2m', img_gray2m)
	plt.figure(35)
	noise_grayHist2m = plt.hist(noise_gray2m.ravel(),256)
	plt.savefig(r'./gamma/noise_grayHist2m.png')
	cv.imshow('img_gray3m', img_gray3m)
	plt.figure(36)
	noise_grayHist3m = plt.hist(noise_gray3m.ravel(), 256)
	plt.savefig(r'./gamma/noise_grayHist3m.png')

	cv.imshow('img_color1m', img_color1m)
	plt.figure(37)
	noise_colorHist1m = plt.hist(noise_color1m.ravel(),256)
	plt.savefig(r'./gamma/noise_colorHist1m.png')
	cv.imshow('img_color2m', img_color2m)
	plt.figure(38)
	noise_colorHist2m = plt.hist(noise_color2m.ravel(),256)
	plt.savefig(r'./gamma/noise_colorHist2m.png')
	cv.imshow('img_color3m', img_color3m)
	plt.figure(39)
	noise_colorHist3m = plt.hist(noise_color3m.ravel(),256)
	plt.savefig(r'./gamma/noise_colorHist3m.png')
	# plt.show()
	cv.waitKey()
	cv.destroyAllWindows()

	# 将加了噪声的图片写入对应的文件夹

	# 原始彩色图像和灰度图像及其直方图

	cv.imwrite(r'./gray_image/The_gray_image.png', gray)

	# guassian_blur()函数进行模糊的图像及其直方图
	for i in range(5):
		cv.imwrite(r'./gaussian_blur/img_gray' + str(i) + 'g.png', a11[i])
		cv.imwrite(r'./gaussian_blur/img_color' + str(i) + 'g.png', a12[i])

	# uGaussianSin_blur()函数进行模糊的图像及其直方图
	for i in range(5):
		cv.imwrite(r'./uGaussianSin_blur/img_gray' + str(i) + 's.png', a21[i])
		cv.imwrite(r'./uGaussianSin_blur/img_color' + str(i) + 's.png', a22[i])

	# uGaussianCos_blur()函数进行模糊的图像及其直方图
	for i in range(5):
		cv.imwrite(r'./uGaussianCos_blur/img_gray' + str(i) + 'c.png', a31[i])
		cv.imwrite(r'./uGaussianCos_blur/img_color' + str(i) + 'c.png', a32[i])

	# gamma()函数进行模糊的图像及其直方图
	for i in range(3):
		cv.imwrite(r'./gamma/img_gray' + str(i) + 'm.png', a41[i])
		cv.imwrite(r'./gamma/img_color' + str(i) + 'm.png', a42[i])







