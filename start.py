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

#os.environ['TZ'] = 'Asia/Shanghai'
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)
from comservice import *

CFG = sys.path[0] + 'config/config.ini'
app = Flask(__name__)

taskLock = Lock()


def __genRPS(response: str):
    """
    generate response data
    :param response:
    :return: Response
    """
    return FL_RP(response, headers={"Server": GV.PROJECT_NAME,
                                    "Life": "Life is a constant struggle."},
                 mimetype=GV.CPU_CONFIG.get("Response", "mimetype"))


@app.route("/runCom", methods={'POST', 'GET'})
def runCom():
    """
    component running controller
    :return:
    """
    response = Response()
    isExtern = False
    comPath = None
    try:
        _params = Common.getParameters(request, response)
        if (Common.checkParameters(_params, "nodeId,comName,args,runType,runSpace")):
            args = _params.get("args")
            nodeId = str(_params.get("nodeId")).lower()
            runMethod = Common.getRunType(_params.get("runMethod"))
            preTrain = False if _params.get("preTrain") is None or _params.get("preTrain").lower() == "false" else True
            author = "" if _params.get("author") is None else _params.get("author")
            runType = _params.get("runType")
            runSpaceDir = _params.get("runSpace").strip()
            response.setRunSpace(runSpaceDir)
            comName, comPath, isExtern = Common.prepareCom(_params)
            if isinstance(args, str):
                args = json.loads(parse.unquote(args))
            if runSpaceDir is None:
                runSpaceDir = GV.CPU_CONFIG.get("RunSpace", "tmpDir")
            else:
                runSpaceDir = str(parse.unquote(runSpaceDir))
            runId = Common.getRunId(comName, nodeId, runType, runSpaceDir)
            if (comName in GV.CPU_COMS):
                taskLock.acquire()
                if isExtern or Common.runSpaceExists(runId) == False:
                    Common.addRunSpace(runId, nodeId=nodeId, comName=comName, comPath=comPath,
                                       runSpaceDir=runSpaceDir)
                if not Common.getRunSpace(runId).isIdle():
                    response.setStatus(ResponseStatus.COM_IN_RUN.getCode())
                    response.setInfo(ResponseStatus.COM_IN_RUN.getMsg())
                    taskLock.release()
                else:
                    Common.setCPUContext(runId, context={"args": _params, "NNContext": NNContext(_params, False)},
                                         preTrain=preTrain)
                    def _run(cpuMonitor=True):
                        Common.getRunSpace(runId).runCom(args=args, response=response, method=runMethod,
                                                         preTrain=preTrain,
                                                         __Author=author, cpuMonitor=cpuMonitor)

                    Common.getRunSpace(runId).runCom(args=args, response=response, method="runCheck",
                                                     preTrain=preTrain)
                    if response.getStatus() == 0:
                        Common.getRunSpace(runId).setReady()
                    taskLock.release()
                    if response.getStatus() == 0:
                        if runMethod == GV.COMPOMENT_RUN_TYPE.distributedRun:
                            _run(cpuMonitor=False)
                            return __genRPS(response.toString())
                        # run with preTrain type
                        if (GV.CPU_CONFIG.get("ENV", "debug").lower() == "true" or preTrain == True):
                            _run(cpuMonitor=False)
                        else:
                            # run with job push type,only return push status,this will create thread to run
                            if Common.cpuProcessExists(runId) == False:
                                Common.initCpuProcess(runId)
                            if Common.getCpuProcess(runId).isIdle():  # need check thread status
                                # if parameters meet the requirements will change to ready
                                Common.getCpuProcess(runId).prepare(threading.Thread(target=_run))
                                response.setInfo("success push task!")
                                response.setKV("logPath", Common.getRunSpace(runId).getLogPath(isDir=False))
                                return __genRPS(response.toString())
                            else:
                                response.setStatus(ResponseStatus.COM_IN_RUN.getCode())
                                response.setInfo(ResponseStatus.COM_IN_RUN.getMsg())

                return __genRPS(response.toString())
            else:
                response.setStatus(ResponseStatus.PARAM_NOT_MATCH.getCode())
                response.setInfo(ResponseStatus.PARAM_NOT_MATCH.getMsg(ext=comName + " compoment not exists!"))
        else:
            response.setStatus(ResponseStatus.PARAM_NOT_SET.getCode())
            response.setInfo(
                ResponseStatus.PARAM_NOT_SET.getMsg(ext="nodeId,comName,args,runType,runSpace not all set!"))

    except Exception as e:
        response.setStatus(ResponseStatus.CODE_EXCEPTON.getCode())
        response.setInfo(Common.formatException(e))
        if isExtern:
            Common.unloadComs(comPath, nodeId)
        if (GV.CPU_CONFIG.get("ENV", "debug").lower() == "true"):
            raise e

    return __genRPS(response.toString())


@app.route("/breakRunning", methods={'POST', 'GET'})
def breakRunning():
    """
    break component running controller
    :return:
    """
    response = Response()
    try:
        _params = Common.getParameters(request, response)
        if (Common.checkParameters(_params, "nodeId,comName,runType,runSpace")):
            nodeId = str(_params.get("nodeId")).lower()
            runType = _params.get("runType")
            comName, comPath, isExtern = Common.prepareCom(_params)
            runSpaceDir = _params.get("runSpace")
            comName = comName.lower()
            response.setRunSpace(runSpaceDir)
            runId = Common.getRunId(comName, nodeId, runType, runSpaceDir)
            if Common.cpuProcessExists(runId):
                Common.getRunSpace(runId).runCom(response=response, method="break")
            else:
                tmpSpace = RunSpace(runId, nodeId=nodeId, comName=comName, comPath=comPath,
                                    runSpaceDir=runSpaceDir)
                if tmpSpace.spaceIsLocked():
                    preTrain = False if _params.get("preTrain") is None or _params.get(
                        "preTrain").lower() == "false" else True
                    Common.setCPUContext(runId, context={"args": _params, "NNContext": NNContext(_params, False)},
                                         preTrain=preTrain)
                    tmpSpace.runCom(response=response, method="break")
                else:
                    response.setStatus(ResponseStatus.COM_NOT_RUN.getCode())
                    response.setInfo(
                        ResponseStatus.COM_NOT_RUN.getMsg())
        else:
            response.setStatus(ResponseStatus.PARAM_NOT_SET.getCode())
            response.setInfo(ResponseStatus.PARAM_NOT_SET.getMsg(ext="nodeId,comName,runType,runSpace not all set!"))

    except Exception as e:
        response.setStatus(ResponseStatus.CODE_EXCEPTON.getCode())
        response.setInfo(Common.formatException(e))
        if (GV.CPU_CONFIG.get("ENV", "debug").lower() == "true"):
            raise e
    finally:
        if isExtern:
            Common.unloadComs(comPath, nodeId)
    return __genRPS(response.toString())

@app.route("/getResponse", methods={'POST', 'GET'})
def getResponse():
    """
    get specific nodeId run response data controller
    :return:
    """
    response = Response()
    try:
        _params = Common.getParameters(request, response)
        if Common.checkParameters(_params, "comName,nodeId,runType,runSpace"):
            nodeId = str(_params.get("nodeId")).lower()
            runType = _params.get("runType")
            runSpaceDir = _params.get("runSpace")
            response.setRunSpace(runSpaceDir)
            comName, comPath, _ = Common.prepareCom(_params)
            comName = comName.lower()
            runId = Common.getRunId(comName, nodeId, runType, runSpaceDir)
            # run thread judge
            if Common.cpuProcessExists(runId):
                if Common.getCpuProcess(runId).isReady():
                    response.setStatus(ResponseStatus.COM_IN_READY.getCode())
                    response.setInfo(ResponseStatus.COM_IN_READY.getMsg())
                    return __genRPS(response.toString())
                elif Common.getCpuProcess(runId).isRunning():
                    response.setStatus(ResponseStatus.DATA_NOT_PREPARE.getCode())
                    response.setInfo(runId + " thread state " + str(Common.getRunSpace(runId).isRunning()) + "," +
                                     ResponseStatus.DATA_NOT_PREPARE.getMsg() + "may need time " +
                                     str(Common.getRunSpace(runId).trackEstimated()) + " s")
                    return __genRPS(response.toString())

            tmpSpace = RunSpace(runId, nodeId=nodeId, comName=comName, comPath=comPath,
                                runSpaceDir=runSpaceDir)
            if (os.path.exists(tmpSpace.getResponsePath())):
                with open(tmpSpace.getResponsePath(), 'r') as f:
                    response.setData(json.loads(f.read()))
            else:  # check RunSpace locker
                if Common.runSpaceExists(runId) == True and Common.getRunSpace(runId).isRunning():
                    response.setStatus(ResponseStatus.DATA_NOT_PREPARE.getCode())
                    response.setInfo(runId + " thread state " + str(Common.getRunSpace(runId).isRunning()) + "," +
                                     ResponseStatus.DATA_NOT_PREPARE.getMsg() + "may need time " +
                                     str(Common.getRunSpace(runId).trackEstimated()) + " s")
                    return __genRPS(response.toString())
                else:
                    if tmpSpace.spaceIsLocked():
                        spaceStatus, _info = tmpSpace.getSpaceStatus()
                        if spaceStatus == GV.COMPOMENT_RUN_STATUS.comExternalInterrupt:
                            response.setStatus(ResponseStatus.EXTERNAL_INTERRUPTED.getCode())
                            response.setInfo(ResponseStatus.EXTERNAL_INTERRUPTED.getMsg() + str(_info))
                            tmpSpace.externallyInterruptFixLog(_info)
                        else:
                            response.setStatus(ResponseStatus.DATA_NOT_PREPARE.getCode())
                            response.setInfo(runId + " is in running," +
                                             ResponseStatus.DATA_NOT_PREPARE.getMsg())
                    else:
                        if tmpSpace.spaceExists():
                            spaceStatus, _info = tmpSpace.getSpaceStatus()
                            if spaceStatus == GV.COMPOMENT_RUN_STATUS.comExternalInterrupt:
                                response.setStatus(ResponseStatus.EXTERNAL_INTERRUPTED.getCode())
                                response.setInfo(ResponseStatus.EXTERNAL_INTERRUPTED.getMsg() + str(_info))
                                tmpSpace.externallyInterruptFixLog(_info)
                            elif spaceStatus == GV.COMPOMENT_RUN_STATUS.comFinish:
                                pass
                            else:
                                response.setStatus(ResponseStatus.UNKNOWN_EXCEPTION.getCode())
                                response.setInfo(
                                    ResponseStatus.UNKNOWN_EXCEPTION.getMsg() + "Componment not run or exception with no relevant output information!")
                        else:
                            response.setStatus(ResponseStatus.UNKNOWN_EXCEPTION.getCode())
                            response.setInfo(
                                ResponseStatus.UNKNOWN_EXCEPTION.getMsg() + "Componment not run or exception with no relevant output information!")
        else:
            response.setStatus(ResponseStatus.PARAM_NOT_SET.getCode())
            response.setInfo(ResponseStatus.PARAM_NOT_SET.getMsg(ext="comName,nodeId,runType,runSpace not set!"))
            return __genRPS(response.toString())
    except Exception as e:
        response.setStatus(ResponseStatus.CODE_EXCEPTON.getCode())
        response.setInfo(Common.formatException(e))
        if (GV.CPU_CONFIG.get("ENV", "debug").lower() == "true"):
            raise e

    return __genRPS(response.toString())


@app.route("/scanComs", methods={'POST', 'GET'})
def scanComs():
    """
    scan all componment lists controller
    :return:
    """
    response = Response()
    try:
        response.setData({"COMS": GV.CPU_COMS})
    except Exception as e:
        response.setStatus(ResponseStatus.CODE_EXCEPTON.getCode())
        response.setInfo(Common.formatException(e))

    return __genRPS(response.toString())


@app.route("/doc", methods={'POST', 'GET'})
def genDoc():
    """
    auto Generate component documents
    :return:
    """
    import importlib
    response = Response()
    tmp = ""
    try:
        res = ""
        css = '<style type="text/css">html{padding:20px} table {font-family: verdana,arial,sans-serif;font-size:12px;color:#333333;border-width: 1px;border-color: #666666;border-collapse: collapse;} table th {text-align:center;border-width: 1px;padding: 8px;border-style: solid;border-color: #666666;background-color: #dedede;} table td {border-width: 1px;padding: 8px;border-style: solid;border-color: #666666;background-color: #ffffff;}</style>'
        url = request.url[:-4]
        for k in GV.CPU_COMS:
            tmp = GV.CPU_COMS[k]
            comDesc = getattr(importlib.import_module(GV.CPU_COMS[k]), Common.getComClassName(k)).info()
            res = "%s\n\n%s" %(res,comDesc.genDoc(GV.CPU_COMS[k][9:].replace("."," > "),url))
        return "<html><head>%s</head><h2>%s Componments Document</h2> Current Version:%s, Total Nums:%s%s<p><hr>©2020 linewell.com</p></html>" % (css,GV.PROJECT_NAME,GV.PROJECT_VERSION,len(GV.CPU_COMS),res)
    except Exception as e:
        response.setStatus(ResponseStatus.CODE_EXCEPTON.getCode())
        response.setInfo(Common.formatException(e)+tmp)
    return __genRPS(response.toString())


@app.route("/comInfo", methods={'POST', 'GET'})
def comInfo():
    """
    get component detail informations controller
    :return:
    """
    response = Response()
    try:
        _params = Common.getParameters(request, response)
        if Common.checkParameters(_params, "comName"):
            comName, comPath, _ = Common.prepareCom(_params)
            if (comName in GV.CPU_COMS):
                RunSpace("", nodeId=GV.DEFAULT_NODE, comName=comName, runSpaceDir="", comPath=comPath).runCom(
                    response=response)
            else:
                response.setStatus(ResponseStatus.PARAM_NOT_MATCH.getCode())
                response.setInfo(ResponseStatus.PARAM_NOT_MATCH.getMsg(ext=comName + " compoment not exists!"))
        else:
            response.setStatus(ResponseStatus.PARAM_NOT_SET.getCode())
            response.setInfo(ResponseStatus.PARAM_NOT_SET.getMsg(ext="comName not set!"))

    except Exception as e:
        response.setStatus(ResponseStatus.CODE_EXCEPTON.getCode())
        response.setInfo(Common.formatException(e))
        if (GV.CPU_CONFIG.get("ENV", "debug").lower() == "true"):
            raise e
    return __genRPS(response.toString())

__methods = {"method": {"comInfo": "Read component detail information.",
                        "breakRunning": "Break component current running state!",
                        "runCom": "Feed Data to run Component.",
                        "scanComs": "list all Components.",
                        "doc": "Get components documents.",
                        "getResponse": "Get the component last run response data."}}

@app.route("/", methods={'POST', 'GET'})
def default():
    response = Response()
    response.setRequest(request.args)
    response.setData(__methods)
    with open(sys.path[0] + '/README.md') as f:
        lines = f.readlines()
    response.setKV("README", lines)
    response.setKV("Environ",str(os.environ))
    return __genRPS(response.toString())


@app.errorhandler(404)
def not_found(error):
    response = Response()
    response.setStatus(ResponseStatus.METHOD_NOT_FOUND.getCode())
    response.setInfo(ResponseStatus.METHOD_NOT_FOUND.getMsg())
    response.setData(__methods)
    return __genRPS(response.toString())


def init_session(conf='config.ini'):
    GV.CPU_CONFIG.read(conf)
    Common.mkdir(GV.CPU_CONFIG.get("ENV", "logPath"))
    Common.sysLog("app init with config " + conf)
    GV.PROJECT_ENV['PATH'] = sys.path[0]
    Common.loadComs(sys.path[0] + "/Core/Com/")


def signalHandler(signum, frame):
    Common.closeCurrentRunningCom()
    Common.sysLog("app are interrupted externally!", level=GV.LOG_LEVEL.WARN)
    os._exit(0)


init_session(CFG)
if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signalHandler)
        signal.signal(signal.SIGTERM, signalHandler)
        Common.sysLog("app start with port " + GV.CPU_CONFIG.get("ENV", "port"))
        os.environ['FLASK_ENV'] = "development" if Common.systemIsDebug() else "product"
        app.run(host='0.0.0.0', port=int(GV.CPU_CONFIG.get("ENV", "port")),
                debug=Common.systemIsDebug())
    except Exception as e:
        Common.sysLog("app start Failed!", level=GV.LOG_LEVEL.ERROR)
        Common.sysLog(Common.formatException(e), level=GV.LOG_LEVEL.ERROR)
        os._exit(0)
else:
    Common.sysLog("app start with outer!")
