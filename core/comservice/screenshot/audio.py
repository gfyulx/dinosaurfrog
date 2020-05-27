#!/usr/bin/env python3
# coding=utf-8

import wave
from pyaudio import PyAudio,paInt16
import time
import datetime
import os

"""
音频处理类
"""
class Audio:
    def __init__(self):
        """
        #NUM_SAMPLES = 2000      # pyAudio内部缓存的块的大小
        #SAMPLING_RATE = 8000    # 取样频率
        #LEVEL = 1500            # 声音保存的阈值
        #COUNT_NUM = 20          # NUM_SAMPLES个取样之内出现COUNT_NUM个大于LEVEL的取样则记录声音
        #SAVE_LENGTH = 8         # 声音记录的最小长度：SAVE_LENGTH * NUM_SAMPLES 个取样
        """
        self.frameRate=8000  #取样频率
        self.level=1500      #声音保存的阈值
        self.numSamples=2000  #pyaudio内部缓存块的大小
        self.countNum=20       #NUM_SAMPLES个取样之内出现COUNT_NUM个大于LEVEL的取样则记录声音
        self.saveLength=2     #声音记录的最小长度：SAVE_LENGTH * NUM_SAMPLES 个取样
        self.channels=1
        #self.sampwidth=8


#麦克风音频录制
    def microPhoneRecord(self,sec,fileDir):
        pa=PyAudio()
        datenow=datetime.datetime.now()
        cc=time.time()
        secsl=(cc-int(cc))*1000
        filePrx="%s/%s%03d"%(fileDir,datenow.strftime("%Y%m%d%H%M%S"),secsl)
        fileName="%s%s"%(filePrx,".wav")
        try:
            stream=pa.open(format = paInt16,channels=self.channels,  #input_device_index为设备索引
                   rate=self.frameRate,input=True,input_device_index=1,
                   frames_per_buffer=self.frameRate)
            my_buf=[]   
            tB=time.time()
            tE=time.time()
            while tE-tB<sec/1000:  #sec为毫秒
                string_audio_data = stream.read(self.frameRate)
                my_buf.append(string_audio_data)
                print('.')
                tE=time.time()
            self.save_wave_file(fileName,my_buf)
        finally:
            stream.close()

    def play(self,fileName):
        chunk=1024  #每次读取时的块数量
        wf=wave.open(fileName,'rb')
        p=PyAudio()
        stream=p.open(format=p.get_format_from_width(wf.getsampwidth()),channels=
        wf.getnchannels(),rate=wf.getframerate(),output=True)
        while True:
            data=wf.readframes(chunk)
            if data=="":break
            stream.write(data)
        stream.close()
        p.terminate()
        

#系统音频录制
    def systemVideoRecord(self,time,fileDir):
        None



    def save_wave_file(self,filename,data):
        '''save the date to the wavfile'''
        wf=wave.open(filename,'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.saveLength)
        wf.setframerate(self.frameRate)
        wf.writeframes(b"".join(data))
        wf.close()

  

if __name__ == '__main__':
    au=Audio()
    au.microPhoneRecord(10000,"./")
    print('Over!') 
    #au.play()