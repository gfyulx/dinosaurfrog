#!/bin/bash
workhome=$(cd $(dirname $0)/../; pwd)
mkdir -p $workhome/logs
nohup python3 $workhome/app.py > $workhome/logs/di.log 2>&1 &
echo $! > $workhome/logs/di.pid
echo "start agent success!"