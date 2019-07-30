#!/usr/bin/env python3
# coding=utf-8

from flask import Flask, request
import time
from threading import Thread

app = Flask(__name__)

def timeSleep(arg):
    print("start",arg)
    time.sleep(1000)
    print("end")
@app.route("/api/test", methods={'POST', 'GET'})
def test():

    pSub=Thread(target=timeSleep,args=("into",))
    pSub.start()
    return "ok"

if __name__=='__main__':
    app.run(host='0.0.0.0', port=18888, debug=True)