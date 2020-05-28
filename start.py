# -*- encoding: utf-8 -*-
"""
@File    :   start.py
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:
@Modify TIme:  2020/5/13 14:46
"""

import os
import sys
import threading
import json
from flask import Flask, request, Response as FL_RP
from urllib import parse
import signal
from threading import Lock
from core.utils import global_variable as GV, common as common
from core.comservice import *

# os.environ['TZ'] = 'Asia/Shanghai'
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

CFG = sys.path[0] + '/config/config.ini'
CFG_PATH = sys.path[0] + "/config"

app = Flask(__name__)

taskLock = Lock()


def __genRPS(response: str):
    """
    generate response data
    :param response:
    :return: Response
    """
    return FL_RP(response, headers={"Server": GV.PROJECT_NAME,
                                    "Description": "a deep learning train and service release framework"},
                 mimetype="application/json")


@app.route("/runCom", methods={'POST', 'GET'})
def runCom():
    """
    run a model com
    :return:
    """
    response = Response()
    try:
        _params = common.get_parameters(request, response)
    except Exception as e:
        response.set_status(ResponseStatus.CODE_EXCEPTON.get_code())
        response.set_info(common.format_exception(e))
        if (GV.CONFIG.get("ENV", "debug").lower() == "true"):
            raise e
    return __genRPS(response.to_string())


@app.route("/breakRunning", methods={'POST', 'GET'})
def breakRunning():
    """
    break running com
    :return:
    """
    response = Response()
    try:
        _params = common.get_parameters(request, response)
    except Exception as e:
        response.set_status(ResponseStatus.CODE_EXCEPTON.get_code())
        response.set_info(common.format_exception(e))
        if (GV.CONFIG.get("ENV", "debug").lower() == "true"):
            raise e
    return __genRPS(response.to_string())


@app.route("/getStatus", methods={'POST', 'GET'})
def getResponse():
    """
    get status of com
    :return:
    """
    response = Response()
    try:
        _params = common.get_parameters(request, response)
    except Exception as e:
        response.set_status(ResponseStatus.CODE_EXCEPTON.get_code())
        response.set_info(common.format_exception(e))
        if (GV.CONFIG.get("ENV", "debug").lower() == "true"):
            raise e
    return __genRPS(response.to_string())



@app.route("/scanComs", methods={'POST', 'GET'})
def scanComs():
    """
    scan all componment lists controller
    :return:
    """
    response = Response()
    try:
        response.set_data({"COMS": GV.COMS})
    except Exception as e:
        response.set_status(ResponseStatus.CODE_EXCEPTON.get_code())
        response.set_info(common.format_exception(e))

    return __genRPS(response.to_string())

@app.route("/comInfo", methods={'POST', 'GET'})
def comInfo():
    """
    get component detail informations controller
    :return:
    """
    response = Response()
    try:
        _params = common.get_parameters(request, response)
    except Exception as e:
        response.set_status(ResponseStatus.CODE_EXCEPTON.get_code())
        response.set_info(common.format_exception(e))
        if (GV.CONFIG.get("ENV", "debug").lower() == "true"):
            raise e
    return __genRPS(response.to_string())


__methods = {"method": {"comInfo": "Read component detail information.",
                        "breakRunning": "Break component current running state!",
                        "runCom": "Feed Data to run Component.",
                        "scanComs": "list all Components.",
                        "getStatus": "Get the component last run response data."}}


@app.route("/", methods={'POST', 'GET'})
def default():
    response = Response()
    response.set_request(request.args)
    response.set_data(__methods)
    with open(sys.path[0] + '/README.md') as f:
        lines = f.readlines()
    response.set_KV("README", lines)
    response.set_KV("Environ", str(os.environ))
    return __genRPS(response.to_string())


@app.errorhandler(404)
def not_found(error):
    response = Response()
    response.set_status(ResponseStatus.METHOD_NOT_FOUND.get_code())
    response.set_info(ResponseStatus.METHOD_NOT_FOUND.get_msg())
    response.set_data(__methods)
    return __genRPS(response.to_string())


def init_session(conf='config/config.ini'):
    print(CFG)
    GV.CONFIG.read(CFG)
    common.mkdir(GV.CONFIG.get("ENV", "logPath"))
    common.sys_log("app init with config " + conf)
    GV.PROJECT_ENV['PATH'] = sys.path[0]



def signal_handler(signum, frame):
    common.close_com()
    common.sys_log("app are interrupted externally!", level=GV.LOG_LEVEL.WARN)
    os._exit(0)


init_session(CFG)
if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        common.sys_log("app start with port " + GV.CONFIG.get("ENV", "port"))
        os.environ['FLASK_ENV'] = "development" if common.system_isdebug() else "product"
        app.run(host='0.0.0.0', port=int(GV.CONFIG.get("ENV", "port")),
                threaded=True,
                debug=common.system_isdebug())
    except Exception as e:
        common.sys_log("app start Failed!", level=GV.LOG_LEVEL.ERROR)
        common.sys_log(common.format_exception(e), level=GV.LOG_LEVEL.ERROR)
        os._exit(0)
else:
    common.sys_log("app start with outer!")
