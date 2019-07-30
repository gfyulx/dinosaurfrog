from multiprocessing import Process
from threading import Thread
import time

def runCom():

    pRun=Process(target=runReal,args=())
    tRun=Thread(target=runReal)
    pRun.start()
    return "end runCom"
    pRun.join()

def runReal():
    print("run runreal")
    time.sleep(20)
    print("runreal end")




if __name__=='__main__':
    print("start call")
    msg=runCom()
    print("===",msg,"===")

    print("end call")