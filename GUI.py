import numpy as np
import cv2 as cv
import wx
import os
import wx.lib.filebrowsebutton
import sys

APP_TITLE = u'美图Beautify'
# APP_ICON = 'res/python.ico'


class mainFrame(wx.Frame):
	'''
	程序主窗口类，继承自wx.Frame
	'''
	wildcard = '*.png|*.png|*.jpg|*.jpg|所有文件(*.*)|*.*'

	def __init__(self, parent):
		wx.Frame.__init__(self, parent, -1, title=APP_TITLE, size=(1000, 600))
		self.ratio = 2
		self.image_i = 1
		self.now_process = 0
		self.panel = wx.Panel(self)
		self.openBtn = wx.Button(self.panel, -1, '打开图像', pos=(10, 20))
		self.openBtn.Bind(wx.EVT_BUTTON, self.OnOpen)
		self.saveAsBtn = wx.Button(self.panel, -1, '将图像另存为', pos=(10, 50))
		self.saveAsBtn.Bind(wx.EVT_BUTTON, self.OnSaveAs)

		self.slider = wx.Slider(self.panel, value=2, minValue=2, maxValue=10, pos=(400, 450), size=(200, 50),
								style=wx.SL_HORIZONTAL | wx.SL_LABELS)
		self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)

		self.CreateStatusBar()
		# 创建菜单栏
		filemenu = wx.Menu()
		func_1 = filemenu.Append(wx.ID_ANY, "半色调实现1", "半色调实现方法一")
		filemenu.AppendSeparator()
		func_2 = filemenu.Append(wx.ID_ANY, "半色调实现2", "半色调实现方法二")
		filemenu.AppendSeparator()
		func_3 = filemenu.Append(wx.ID_ANY, "肤色调整", "肤色调整")
		filemenu.AppendSeparator()
		func_4 = filemenu.Append(wx.ID_ANY, "祛斑/祛痘", "祛斑/祛痘")
		filemenu.AppendSeparator()
		func_5 = filemenu.Append(wx.ID_ANY, "去毛孔", "去毛孔")
		filemenu.AppendSeparator()
		func_6 = filemenu.Append(wx.ID_ANY, "贴图", "贴图")
		filemenu.AppendSeparator()
		exit = filemenu.Append(wx.ID_EXIT, "退出", " 退出程序")
		self.Bind(wx.EVT_MENU, self.Onfunc_1, func_1)
		self.Bind(wx.EVT_MENU, self.Onfunc_2, func_2)
		self.Bind(wx.EVT_MENU, self.Onfunc_3, func_3)
		self.Bind(wx.EVT_MENU, self.Onfunc_4, func_4)
		self.Bind(wx.EVT_MENU, self.Onfunc_5, func_5)
		self.Bind(wx.EVT_MENU, self.Onfunc_6, func_6)
		self.Bind(wx.EVT_MENU, self.OnExit, exit)

		######################################
		filemenu_2 = wx.Menu()
		about = filemenu_2.Append(wx.ID_ABOUT, "相关信息", "相关信息")
		filemenu_2.AppendSeparator()
		help = filemenu_2.Append(wx.ID_ANY, "帮助", "帮助")
		self.Bind(wx.EVT_MENU, self.OnAbout, about)
		self.Bind(wx.EVT_MENU, self.Onhelp, help)

		menuBar = wx.MenuBar()
		menuBar.Append(filemenu, "开始")
		menuBar.Append(filemenu_2, "关于")
		self.SetMenuBar(menuBar)
		self.Show(True)

	def OnOpen(self, event):
		self.now_process = 1
		dlg = wx.FileDialog(self, message='打开文件',
							defaultDir='',
							defaultFile='',
							wildcard=self.wildcard,
							style=wx.FD_OPEN)
		if dlg.ShowModal() == wx.ID_OK:
			self.filePath = dlg.GetPath()

			image = wx.Image(self.filePath, wx.BITMAP_TYPE_ANY)
			image.Rescale(300, 300)
			bitpic = image.ConvertToBitmap()
			wx.StaticBitmap(self.panel, -1, bitmap=bitpic, pos=(100, 100))
			# wx.StaticBitmap(self.panel, -1, bitmap=bitpic, pos=(600, 100))
			dlg.Destroy()

	def OnSaveAs(self, event):
		if self.now_process != 0:
			dlg = wx.FileDialog(self, '另存为', os.getcwd(),
								defaultFile='',
								style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
								wildcard=self.wildcard
								)
			if dlg.ShowModal() == wx.ID_OK:
				file_p = dlg.GetPath()
				print(file_p)
				cv.imwrite(file_p, self.image)
				save_message = wx.MessageDialog(self, '图像已保存', '提示')
				save_message.ShowModal()
				save_message.Destroy()
				dlg.Destroy()
			else:
				mes2 = wx.MessageDialog(self, '未打开图像，请先打开图像！', '提示')
				mes2.ShowModal()
				mes2.Destroy()


	def OnAbout(self, e):
		# A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
		dlg1 = wx.MessageDialog(self, "该软件由华南理工大学蔡翔、成明、廖昊旻、宋皓楠、谌浩开发。\n"
									 "主要功能有：\n两种图像半色调算法的实现以及"
									 "美颜相机具备的基本功能（例如肤色调整、祛斑/祛痘、去毛孔、贴图等等）",
							   "相关信息", wx.OK)
		dlg1.ShowModal()  # Show it
		dlg1.Destroy()  # finally destroy it when finished.

	def OnExit(self, e):
		self.Close(True)

	def Onhelp(self, event):
		# A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
		dlg2 = wx.MessageDialog(self, "该软件主要功能及使用方法如下所述：\n"
									 "半色调处理——方案一：\n"
									  "半色调处理——方案二：\n"
									  "肤色调整：\n"
									  "祛斑/祛痘：\n"
									  "去毛孔：\n"
									  "贴图：\n"
									  "其他：\n"
							   ,"帮助", wx.OK)
		dlg2.ShowModal()  # Show it
		dlg2.Destroy()  # finally destroy it when finished.

	def OnSliderScroll(self, event):
		self.ratio = self.slider.GetValue()
		if self.now_process == 3:
			self.Onfunc_3(self)
		print(" ")

	def Onfunc_1(self, event):
		"""
		半色调技术--方法一
		:param event:
		:return:
		"""

		print(" ")

	def Onfunc_2(self, event):
		"""
		半色调技术--方法二
		:param event:
		:return:
		"""
		print(" ")

	def Onfunc_3(self, event):
		"""
		肤色调整
		:param event:
		:return:
		"""
		if self.now_process != 0:
			self.now_process = 3
			image = cv.imread(self.filePath)
			img_0 = image.copy()
			img_0 = np.array(img_0) / 255
			# print(img_0)
			series = img_0.copy()
			height, width = image.shape[0], image.shape[1]
			for i in range(height):
				for j in range(width):
					series[i][j] = np.log(np.multiply(img_0[i][j], (self.ratio - 1)) + 1) / np.log(self.ratio)####
			img_0 = np.multiply(255, series)
			img_0 = np.rint(img_0)
			img_0 = np.array(img_0, dtype=np.uint8)
			self.image = img_0
			cv.imwrite(r'.\new_0.png', img_0)

			img_1 = wx.Image(r'.\new_0.png', wx.BITMAP_TYPE_ANY)
			img_1.Rescale(300, 300)
			bitpic_1 = img_1.ConvertToBitmap()
			wx.StaticBitmap(self.panel, -1, bitmap=bitpic_1, pos=(600, 100))
			os.remove(r'.\new_0.png')
			print(" ")
		else:
			mes1 = wx.MessageDialog(self, '未打开图像，请先打开图像！', '提示')
			mes1.ShowModal()
			mes1.Destroy()

	def Onfunc_4(self, event):
		"""
		祛斑/祛痘
		:param event:
		:return:
		"""
		print(" ")

	def Onfunc_5(self, event):
		"""
		去毛孔
		:param event:
		:return:
		"""
		print(" ")

	def Onfunc_6(self, event):
		"""
		贴图
		:param event:
		:return:
		"""
		print(" ")


if __name__ == "__main__":
	app = wx.App(False)
	frame = mainFrame(None)
	app.MainLoop()


