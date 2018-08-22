#!/usr/bin/env python3
# coding=utf-8

import win32gui
import win32api
import win32con
import win32ui
import time
import datetime
from PIL import Image
import os
import numpy as np
import cv2

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

    def WinCaptureToVideo(self,handle,secs,fileDir):       
        fps=int(1000/secs)
        #fps=25
        l,t,r,b=self.getWinRect(handle)
        fourcc=cv2.VideoWriter_fourcc('X','V','I','D')
        datenow=datetime.datetime.now()
        cc=time.time()
        secsl=(cc-int(cc))*1000
        filePrx="%s/%d_%s%03d"%(fileDir,handle,datenow.strftime("%Y%m%d%H%M%S"),secsl)
        videoFile="%s%s"%(filePrx,".avi")
        video=cv2.VideoWriter(videoFile,fourcc,fps,(r-l,b-t))
        """
        """
        hwnd = handle
        hwndDC = win32gui.GetWindowDC(hwnd)  
        mfcDC=win32ui.CreateDCFromHandle(hwndDC)  
        saveDC=mfcDC.CreateCompatibleDC()  
        saveBitMap = win32ui.CreateBitmap()  
        l,t,r,b=self.getWinRect(handle)
        w=r-l
        h=b-t
        saveBitMap.CreateCompatibleBitmap(mfcDC, w,h)
        saveDC.SelectObject(saveBitMap)  
        #saveDC.BitBlt((0,0),(w, h) , mfcDC, (l,t), win32con.SRCCOPY)   
        
        try:
            while True:
                saveDC.BitBlt((0,0),(w, h) , mfcDC, (0,0), win32con.SRCCOPY)
                datenow=datetime.datetime.now()
                filePrx="%s/%d"%(fileDir,handle)
                bmpname=filePrx+".bmp"       
                saveBitMap.SaveBitmapFile(saveDC, bmpname)
                pic = Image.open(bmpname)
                picNew=pic.crop((0,0,w,h))
                os.remove(bmpname)
                #return picNew  #如果是视频模式，直接返回
                #pic=self.WinCapture(handle,fileDir)
                imm=cv2.cvtColor(np.array(picNew), cv2.COLOR_RGB2BGR)
                video.write(imm)
                #pic.close() 
                #win32gui.DeleteObject(handle)
                time.sleep(0.01)               
        except Exception as e:
            print(e)
        finally:
            video.release()
                    
    #截屏并保存
    #间隔指定时间截屏
    #默认的文件名保存为handle_time.jepg;
    #time为YYYYmmddHHMMDDSSsss;
    def WinCaptureByTime(self,handle,secs,fileDir,saveFlag=1):
        while True:
            self.WinCapture(handle,fileDir,saveFlag)
            time.sleep(secs/1000)        

    def WinCapture(self,handle,fileDir,saveFlag=0):
        hwnd = handle
        hwndDC = win32gui.GetWindowDC(hwnd)  
        mfcDC=win32ui.CreateDCFromHandle(hwndDC)  
        saveDC=mfcDC.CreateCompatibleDC()  
        saveBitMap = win32ui.CreateBitmap()  
        l,t,r,b=self.getWinRect(handle)
        w=r-l
        h=b-t
        saveBitMap.CreateCompatibleBitmap(mfcDC, w,h)
        saveDC.SelectObject(saveBitMap)  
        #saveDC.BitBlt((0,0),(w, h) , mfcDC, (l,t), win32con.SRCCOPY)
        saveDC.BitBlt((0,0),(w, h) , mfcDC, (0,0), win32con.SRCCOPY)
        datenow=datetime.datetime.now()
        cc=time.time()
        secs=(cc-int(cc))*1000
        filePrx="%s/%d_%s%03d"%(fileDir,handle,datenow.strftime("%Y%m%d%H%M%S"),secs)
        bmpname=filePrx+".bmp"       
        saveBitMap.SaveBitmapFile(saveDC, bmpname)
        pic = Image.open(bmpname)
        if saveFlag==0:
            picNew=pic.crop((0,0,w,h))
            os.remove(bmpname)
            return picNew  #如果是视频模式，直接返回
        fileName=filePrx+".jpeg"
        pic.save(os.path.join(fileDir,fileName), 'jpeg')
        os.remove(bmpname)
        return pic
    #查找指定句柄的第N个子窗口
    def findSubWinX(self,hadnle,idx):
        if not handle:         
            return      
        hwndChildList = []     
        win32gui.EnumChildWindows(handle, lambda hwnd, param: param.append(hwnd),  hwndChildList)
        if len(hwndChildList)>=idx and idx>=1:
            return hwndChildList[idx-1]
        else:
            return None  

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
        """ 相对桌面
        left:窗口的左上坐标
        top:窗口的顶部坐标
        right:窗口的右上相对坐标
        bottom:窗口的右下坐标
        """
        if handle>0:
            return win32gui.GetWindowRect(handle)
            #return win32gui.GetClientRect(handle)            
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


   

