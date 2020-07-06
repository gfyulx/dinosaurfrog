# -*- encoding: utf-8 -*-
"""
@File    :   pyc_deploy.py    
@Author  :   gfyulx@163.com
@Version :    1.0
@Description:
@Modify TIme:  2020/7/6 16:19
"""

import os
import shutil
from py_compile import compile

def pyc_ompile(cmd,path):
    comd = cmd  #输入的命令
    path = path  #文件的地址
    if os.path.exists(path) and os.path.isdir(path):
        for parent,dirname,filename in os.walk(path):
            for cfile in filename:
                fullname = os.path.join(parent,cfile)
                if comd == 'clean' and cfile[-4:] == '.pyc':
                    try:
                        os.remove(fullname)
                        print("Success remove file:%s" % fullname)
                    except:
                        print("Can't remove file:%s" % fullname)
                if comd == 'compile' and cfile[-3:] == '.py':
                    try:
                        compile(fullname)
                        print("Success compile file:%s" % fullname)
                    except:
                        print("Can't compile file:%s" % fullname)
                if comd == 'remove' and cfile[-3:] == '.py' and cfile != 'settings.py' and cfile != 'wsgi.py':
                    try:
                        os.remove(fullname)
                        print("Success remove file:%s" % fullname)
                    except:
                        print("Can't remove file:%s" % fullname)
                if comd=='copy' and cfile[-4:] == '.pyc':
                    parent_list = parent.split("/")[:-1]
                    parent_up_path = ''
                    for i in range(len(parent_list)):
                        parent_up_path+=parent_list[i]+'/'
                    shutil.copy(fullname,parent_up_path)
                if comd=='cpython' and cfile[-4:] =='.pyc':
                    cfile_name = ''
                    cfile_list = cfile.split('.')
                    for i in range(len(cfile_list)):
                        if cfile_list[i]=='cpython-36':
                            continue
                        cfile_name+=cfile_list[i]
                        if i==len(cfile_list)-1:
                            continue
                        cfile_name+='.'
                    shutil.move(fullname,os.path.join(parent,cfile_name))

    else:
        print("Not an directory or Direcotry doesn't exist!")


if __name__=='__main__':
    pyc_ompile(compile,"./")
