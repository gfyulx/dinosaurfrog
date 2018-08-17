#!/usr/bin/env python3
# coding=utf-8

import win32gui as win32
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

    def findWindByName(self, ClassName, WinName):
        handle = win32.FindWindow(ClassName, WinName)
        return handle
    """
    根据句柄获取类名，方便下次查找
    """

    def getClassNameByHandle(self, pHandle):
        className = win32.GetClassName(pHandle)
        return className

    def getTitleByHandle(self, pHandle):
        winText = win32.GetWindowText(pHandle)
        return winText
    """
    获取当前活动窗口句柄
    """

    def getForegroundWindow(self):
        currHandle = win32.GetForegroundWindow()
        return currHandle


if __name__ == "__main__":
    screenShotObj = ScreenShot()
    handle = screenShotObj.findWindByName(None, "网易有道词典")
    print("%0x" % handle)
    print("%0x" % screenShotObj.getForegroundWindow())
    print(screenShotObj.getTitleByHandle(handle))
    print(screenShotObj.getClassNameByHandle(handle))
    screenShotObj.getScreenSize()
    screenShotObj.setAttr(handle)
