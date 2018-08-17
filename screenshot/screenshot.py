#!/usr/bin/env python3
# coding=utf-8

import win32gui
import win32api


class ScreenShot(object):
    def __init__(self):
        self.className = None
        self.handle = 0
        self.winText = None
        self.currHandle = 0
        self.pcoor = {'lx': 0, 'ly': 0, 'rx': 0,
                      'ry': 0}  # 窗口的物理坐标，左上x,左上y,右下x,右下y
        self.lcoor = {'lx': 0, 'ly': 0, 'rx': 0,
                      'ry': 0}  # 窗口的逻辑坐标，左上x,左上y,右下x,右下y
        self.screenSize = None
        self.fullScreen = 0  # 是否全屏

    def setAttr(self, handle):
        self.className = self.getClassNameByHandle(handle)
        self.winText = self.getTitleByHandle(handle)
        self.currHandle = self.getForegroundWindow()
        self.screenSize = self.getScreenSize()
        self.checkFullScreen()
        self.lcoor['lx'],self.lcoor['ly'],self.lcoor['rx'],self.lcoor['ry']=self.getWinRect(handle)

    def checkFullScreen(self):
        if (self.pcoor['rx']-self.pcoor['lx'] == self.screenSize['width']) \
            and (self.pcoor['ry']-self.pcoor['ly'] == self.screenSize['high']):
            self.fullScreen = 1
        else:
            self.fullScreen=0

    def getScreenSize(self):
        width = win32api.GetSystemMetrics(0)
        high = win32api.GetSystemMetrics(1)
        return {'width': width, 'high': high}

    #按类或者窗口名称返回窗口的主句柄
    def findWindByName(self, ClassName, WinName):
        handle = win32gui.FindWindow(ClassName, WinName)
        handleParent=win32gui.GetParent(handle)
        if handleParent==0:
            return handle
        else:
            return handleParent
    """
    根据句柄获取类名，方便下次查找
    """

    def getClassNameByHandle(self, pHandle):
        if pHandle>0:
            className = win32gui.GetClassName(pHandle)
        else:
            className=None
        return className

    def getTitleByHandle(self, pHandle):
        winText = win32gui.GetWindowText(pHandle)
        return winText
    """
    获取当前活动窗口句柄
    """

    def getForegroundWindow(self):
        currHandle = win32gui.GetForegroundWindow()
        return currHandle
    def getWinRect(self,handle):
        """
        left:窗口的左上坐标
        top:窗口的顶部坐标
        right:窗口的右上相对坐标
        bottom:窗口的右下坐标
        """
        if handle>0:
            return win32gui.GetWindowRect(handle)
        else:
            return 0,0,0,0
        
    #获取鼠标所在的坐标
    def getMousePoint(self):
        return win32gui.GetCursorPos()
    #获取指定坐标点的窗口句柄
    def getHandleByPoint(self,point):
        try:
            handle=win32gui.WindowFromPoint(point)
        except Exception as e:
            handle=0
        return handle


