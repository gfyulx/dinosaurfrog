#!/bin/bash
workdir=$(cd $(dirname $0); pwd)
workhome=$workdir/../
pid=$(cat $workhome/logs/di.pid)
kill -9 $pid
echo "stop di success!"