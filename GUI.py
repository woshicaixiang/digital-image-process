# -*- coding: utf-8 -*-
"""
GUI.py 美图APP主程序

@author: Xiang Cai, Ming Cheng, Haomin Liao

South China University of Technology

Python Version = 3.7

version of the Python packages:
numpy 1.17.4, OpenCV-Python 4.1.2.30, wxPython 4.1.0
"""
import numpy as np
import numpy.matlib
import cv2 as cv
import wx
import os
import wx.lib.filebrowsebutton
from image_capture import camera
from emboss_function import emboss

APP_TITLE = u'美图Beautify'


class MainFrame(wx.Frame):
    """程序主窗口类，继承自wx.Frame"""
    wildcard = '*.png|*.png|*.jpg|*.jpg|所有文件(*.*)|*.*'

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, title=APP_TITLE, size=(1000, 600))
        self.ratio = 2
        self.image_i = 1
        self.imgFlag = 0
        self.sliderFlag = False
        self.panel = wx.Panel(self)

        # 指定新字体的静态文本
        str1 = "原始图像预览"
        str2 = "处理后的图像预览"
        text_1 = wx.StaticText(self.panel, -1, str1, (100, 50))
        font = wx.Font(18, wx.DECORATIVE, wx.ITALIC, wx.NORMAL)
        text_1.SetFont(font)
        text_1.SetForegroundColour('black')
        text_1.SetBackgroundColour('white')
        text_2 = wx.StaticText(self.panel, -1, str2, (600, 50))
        text_2.SetFont(font)
        text_2.SetForegroundColour('black')
        text_2.SetBackgroundColour('white')

        # 创建菜单栏
        self.CreateStatusBar()
        filemenu = wx.Menu()
        func_1 = filemenu.Append(wx.ID_ANY, "拍摄", "从相机获取图像")
        filemenu.AppendSeparator()
        func_2 = filemenu.Append(wx.ID_ANY, "打开图像", "从电脑获取现有图像")
        filemenu.AppendSeparator()
        func_3 = filemenu.Append(wx.ID_ANY, "保存图像", "保存更改后的图像")
        filemenu.AppendSeparator()
        func_5 = filemenu.Append(wx.ID_ANY, "磨皮", "磨皮")
        filemenu.AppendSeparator()
        # 创建美白子菜单的两个菜单项
        white_menu = wx.Menu()
        white_item0 = white_menu.Append(wx.ID_ANY, "美白（无肤色检测）", "美白（无肤色检测）")
        white_menu.AppendSeparator()
        white_item1 = white_menu.Append(wx.ID_ANY, "美白（含肤色检测）", "美白（含肤色检测）")
        white_menu.AppendSeparator()
        # 将美白子菜单添加到开始菜单中
        filemenu.AppendSubMenu(white_menu, "美白", "八向浮雕 | 调和浮雕")
        filemenu.AppendSeparator()
        # 创建浮雕子菜单的两个菜单项
        emboss_menu = wx.Menu()
        emboss_item0 = emboss_menu.Append(wx.ID_ANY, "八向浮雕", "八向浮雕")
        emboss_menu.AppendSeparator()
        emboss_item1 = emboss_menu.Append(wx.ID_ANY, "调和浮雕", "调和浮雕")
        emboss_menu.AppendSeparator()
        # 将浮雕子菜单添加到开始菜单中
        filemenu.AppendSubMenu(emboss_menu, "浮雕效果", "八向浮雕 | 调和浮雕")
        filemenu.AppendSeparator()
        exit = filemenu.Append(wx.ID_EXIT, "退出", " 退出程序")
        self.Bind(wx.EVT_MENU, self.Onfunc_1, func_1)
        self.Bind(wx.EVT_MENU, self.Onfunc_2, func_2)
        self.Bind(wx.EVT_MENU, self.Onfunc_3, func_3)
        self.Bind(wx.EVT_MENU, self.Onfunc_4_1, white_item0)
        self.Bind(wx.EVT_MENU, self.Onfunc_4_2, white_item1)
        self.Bind(wx.EVT_MENU, self.Onfunc_5, func_5)
        self.Bind(wx.EVT_MENU, self.emboss_func0, emboss_item0)
        self.Bind(wx.EVT_MENU, self.emboss_func1, emboss_item1)
        self.Bind(wx.EVT_MENU, self.OnExit, exit)

        # 创建菜单栏
        filemenu2 = wx.Menu()
        about = filemenu2.Append(wx.ID_ABOUT, "相关信息", "相关信息")
        filemenu2.AppendSeparator()
        help = filemenu2.Append(wx.ID_ANY, "帮助", "帮助")
        self.Bind(wx.EVT_MENU, self.OnAbout, about)
        self.Bind(wx.EVT_MENU, self.Onhelp, help)
        menuBar = wx.MenuBar()
        menuBar.Append(filemenu, "开始")
        menuBar.Append(filemenu2, "关于")
        self.SetMenuBar(menuBar)
        self.Show(True)

    def OnAbout(self, e):
        # 相关信息
        dlg1 = wx.MessageDialog(self, "该软件由华南理工大学蔡翔、成明、廖昊旻、宋皓楠、谌浩开发。\n"
                                      "主要功能有：\n三种图像半色调算法的实现以及"
                                      "美颜相机具备的基本功能（美白、磨皮、浮雕效果等等）",
                                "相关信息", wx.OK)
        dlg1.ShowModal()  # Show it
        dlg1.Destroy()  # finally destroy it when finished.

    def OnExit(self, e):
        """
        退出
        :param e:
        :return:
        """
        self.Close(True)

    def Onhelp(self, event):
        # A message dialog box with an OK button. wx.OK is a standard ID in wxWidgets.
        dlg2 = wx.MessageDialog(self, "该软件主要功能及相关说明如下所述：\n"
                                      "首先需要打开一张图片或者拍摄一张图片，再进行以下操作："
                                      "磨皮、美白、浮雕效果。\n"
                                      "备注：\n"
                                      "其中美白和磨皮处理可以在互相处理的基础上再处理，且二者都可以用滑块控制美白以及磨皮的程度，\n"
                                      "而浮雕效果则是独立开来处理原始图像。\n"
                                , "帮助", wx.OK)
        dlg2.ShowModal()  # Show it
        dlg2.Destroy()  # finally destroy it when finished.

    def surface_blur(self, image_in, thre, blur_radius):
        image_out = image_in * 1.0
        row, col = image_in.shape
        w_size = blur_radius * 2 + 1
        for ii in range(blur_radius, row - 1 - blur_radius):
            for jj in range(blur_radius, col - 1 - blur_radius):
                aa = image_in[ii - blur_radius:ii + blur_radius + 1, jj - blur_radius:jj + blur_radius + 1]
                p0 = image_in[ii, jj]
                mask_1 = np.matlib.repmat(p0, w_size, w_size)
                mask_2 = 1 - abs(aa - mask_1) / (2.5 * thre);
                mask_3 = mask_2 * (mask_2 > 0)
                t1 = aa * mask_3
                image_out[ii, jj] = t1.sum() / mask_3.sum()

        return image_out

    def rgb_surface_blur(self, img, thre, radius):
        img_out = img * 1.0
        thre = thre
        blur_radius = radius
        img_out[:, :, 0] = self.surface_blur(img[:, :, 0], thre, blur_radius)
        img_out[:, :, 1] = self.surface_blur(img[:, :, 1], thre, blur_radius)
        img_out[:, :, 2] = self.surface_blur(img[:, :, 2], thre, blur_radius)

        return img_out

    def Onfunc_1(self, event):
        """从摄像头捕获图像，使用摄像头时按下esc退出，按下空格键进行拍照"""
        if self.sliderFlag:
            self.slider.Destroy()
            self.sliderFlag = False
        img_camera = camera()
        if img_camera is not None:
            dlg = wx.FileDialog(self, '另存为', os.getcwd(),
                                defaultFile='',
                                style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                wildcard=self.wildcard
                                )
            if dlg.ShowModal() == wx.ID_OK:
                self.imgFlag = 1
                self.filePath = dlg.GetPath()
                print(self.filePath)
                cv.imwrite(self.filePath, img_camera)
                save_message = wx.MessageDialog(self, '图像已保存', '提示')
                save_message.ShowModal()
                save_message.Destroy()
                dlg.Destroy()
                self.image = img_camera
                # cv.imwrite(self.filePath, img_camera)
                image = wx.Image(self.filePath, wx.BITMAP_TYPE_ANY)
                image.Rescale(300, 300)
                bitpic = image.ConvertToBitmap()
                wx.StaticBitmap(self.panel, -1, bitmap=bitpic, pos=(100, 100))

    def Onfunc_2(self, event):
        """打开图像"""
        if self.sliderFlag:
            self.slider.Destroy()
            self.sliderFlag = False
        dlg = wx.FileDialog(self, message='打开文件',
                            defaultDir='',
                            defaultFile='',
                            wildcard=self.wildcard,
                            style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.imgFlag = 2
            self.filePath = dlg.GetPath()
            self.image = cv.imread(self.filePath)
            image = wx.Image(self.filePath, wx.BITMAP_TYPE_ANY)
            image.Rescale(300, 300)
            bitpic = image.ConvertToBitmap()
            wx.StaticBitmap(self.panel, -1, bitmap=bitpic, pos=(100, 100))
            # wx.StaticBitmap(self.panel, -1, bitmap=bitpic, pos=(600, 100))
            dlg.Destroy()

    def Onfunc_3(self, event):
        """保存更改后的图像"""
        if self.sliderFlag:
            self.slider.Destroy()
            self.sliderFlag = False
        if self.imgFlag != 0:
            self.imgFlag = 3
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

    def Onfunc_4_1(self, event):
        """美白，未使用肤色检验"""
        if self.imgFlag != 0:
            if self.imgFlag == 6:
                pass
            else:
                self.image = cv.imread(self.filePath)

            self.slider = wx.Slider(self.panel, value=2, minValue=2, maxValue=10, pos=(400, 450), size=(200, 50),
                                    style=wx.SL_HORIZONTAL | wx.SL_LABELS)
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.sliderFlag = True
            self.white_1()
        else:
            mes1 = wx.MessageDialog(self, '未打开图像，请先打开图像！', '提示')
            mes1.ShowModal()
            mes1.Destroy()

    def Onfunc_4_2(self, event):
        """美白，使用了肤色检验"""
        if self.imgFlag != 0:
            if self.imgFlag == 6:
                pass
            else:
                self.image = cv.imread(self.filePath)

            self.slider = wx.Slider(self.panel, value=2, minValue=2, maxValue=10, pos=(400, 450), size=(200, 50),
                                    style=wx.SL_HORIZONTAL | wx.SL_LABELS)
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.sliderFlag = True
            self.white_2()
        else:
            mes1 = wx.MessageDialog(self, '未打开图像，请先打开图像！', '提示')
            mes1.ShowModal()
            mes1.Destroy()

    def Onfunc_5(self, event):
        """磨皮"""
        if self.imgFlag != 0:
            if self.imgFlag == 4 or self.imgFlag == 5:
                pass
            else:
                self.image = cv.imread(self.filePath)
            self.slider = wx.Slider(self.panel, value=2, minValue=2, maxValue=10, pos=(400, 450), size=(200, 50),
                                    style=wx.SL_HORIZONTAL | wx.SL_LABELS)
            self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
            self.sliderFlag = True
            self.imgFlag = 6
            self.skin_process()

        else:
            mes1 = wx.MessageDialog(self, '未打开图像，请先打开图像！', '提示')
            mes1.ShowModal()
            mes1.Destroy()

    def emboss_func0(self, event):
        """浮雕效果(八向浮雕)"""
        if self.sliderFlag:
            self.slider.Destroy()
            self.sliderFlag = False
        if self.imgFlag != 0:
            self.imgFlag = 7
            self.image = cv.imread(self.filePath)
            new_img = emboss(self.image, emboss_type=0)
            cv.imwrite(r'.\temp.png', new_img)

            img_1 = wx.Image(r'.\temp.png', wx.BITMAP_TYPE_ANY)
            img_1.Rescale(300, 300)
            bitpic_1 = img_1.ConvertToBitmap()
            wx.StaticBitmap(self.panel, -1, bitmap=bitpic_1, pos=(600, 100))
            os.remove(r'.\temp.png')
        else:
            mes1 = wx.MessageDialog(self, '未打开图像，请先打开图像！', '提示')
            mes1.ShowModal()
            mes1.Destroy()

    def emboss_func1(self, event):
        """浮雕效果(调和浮雕)"""
        if self.sliderFlag:
            self.slider.Destroy()
            self.sliderFlag = False
        if self.imgFlag != 0:
            self.imgFlag = 8
            self.image = cv.imread(self.filePath)
            new_img = emboss(self.image, emboss_type=1)
            cv.imwrite(r'.\temp.png', new_img)

            img_1 = wx.Image(r'.\temp.png', wx.BITMAP_TYPE_ANY)
            img_1.Rescale(300, 300)
            bitpic_1 = img_1.ConvertToBitmap()
            wx.StaticBitmap(self.panel, -1, bitmap=bitpic_1, pos=(600, 100))
            os.remove(r'.\temp.png')
        else:
            mes1 = wx.MessageDialog(self, '未打开图像，请先打开图像！', '提示')
            mes1.ShowModal()
            mes1.Destroy()

    def OnSliderScroll(self, event):
        """
        滑块触发
        :param event:
        :return:
        """
        self.ratio = self.slider.GetValue()
        if self.imgFlag == 4:
            self.white_1()
        if self.imgFlag == 5:
            self.white_2()
        if self.imgFlag == 6:
            self.skin_process()

    def cr_otsu(self):
        """
		肤色检测
		YCrCb颜色空间的Cr分量+Otsu阈值分割
		"""
        ycrcb = cv.cvtColor(self.image, cv.COLOR_BGR2YCR_CB)
        (y, cr, cb) = cv.split(ycrcb)
        cr1 = cv.GaussianBlur(cr, (5, 5), 0)
        _, self.skin = cv.threshold(cr1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # 闭操作
        self.skin = cv.morphologyEx(self.skin, cv.MORPH_CLOSE, kernel, iterations=5)
        # 膨胀
        self.skin = cv.dilate(self.skin, kernel, 5)

    def white_1(self):
        """
		美白，没有用肤色检测
		"""
        if self.imgFlag == 6:
            pass
        else:
            self.image = cv.imread(self.filePath)
        self.imgFlag = 4
        img_0 = self.image.copy()
        img_0 = np.array(img_0) / 255
        # print(img_0)
        series = img_0.copy()
        height, width = self.image.shape[0], self.image.shape[1]
        for i in range(height):
            for j in range(width):
                series[i][j] = np.log(np.multiply(img_0[i][j], (self.ratio - 1)) + 1) / np.log(self.ratio)
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

    def white_2(self):
        """
        美白，使用了肤色检测
        """
        if self.imgFlag == 6:
            pass
        else:
            self.image = cv.imread(self.filePath)
        self.imgFlag = 5
        img_0 = self.image.copy()
        self.cr_otsu()
        img_0 = np.array(img_0) / 255
        # print(self.skin)
        series = img_0.copy()
        height, width = self.image.shape[0], self.image.shape[1]
        for i in range(height):
            for j in range(width):
                if self.skin[i, j] == 255:
                    series[i][j] = np.log(np.multiply(img_0[i][j], (self.ratio - 1)) + 1) / np.log(self.ratio)
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

    def skin_process(self):
        """
        磨皮
        :return:
        """
        if self.imgFlag == 4 or self.imgFlag == 5:
            pass
        else:
            self.image = cv.imread(self.filePath)
        self.imgFlag = 6
        img_0 = self.image
        img_0 = self.rgb_surface_blur(img_0, 20, self.ratio * 2)
        img_0 = np.array(img_0, dtype=np.uint8)
        self.image = img_0
        cv.imwrite(r'.\new_0.png', img_0)
        img_1 = wx.Image(r'.\new_0.png', wx.BITMAP_TYPE_ANY)
        img_1.Rescale(300, 300)
        bitpic_1 = img_1.ConvertToBitmap()
        wx.StaticBitmap(self.panel, -1, bitmap=bitpic_1, pos=(600, 100))
        os.remove(r'.\new_0.png')

if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame(None)
    app.MainLoop()
