#!/usr/bin/env python3
# coding=utf-8
import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from os.path import join as path_join

print("Remember: Edit at first ffmpegpath within setup.py")

##################################################################################
try:
  import numpy.distutils.misc_util as nd
  with_numpy=True
except:
  with_numpy=False
  sys.stderr.write("NumPy does not seems to be installed on your system.\n")
  sys.stderr.write("You may still use PyFFmpeg, but audio support and NumPy bridge are disabled.\n")  

##################################################################################
## Try to locate source if necessary
if sys.platform in [ 'win32', 'win64' ] :
    ffmpegpath = r'c:\ffmpeg'
    for x in [ r'..\libs\ffmpeg']:
        try:
             os.stat(x)
             ffmpegpath = x
        except:
            pass
    extra_compiler_args=["-static-libgcc"]
    

##################################################################################
# Try to resove
# static dependencies resolution by looking into pkgconfig files
def static_resolver(libs):
    deps = []
    for lib in libs:
        try:
            pc = open(path_join(ffmpegpath, 'lib', 'pkgconfig', 'lib' + lib + '.pc'))
        except IOError:
            continue

        # we only need line starting with 'Libs:'
        l = list(filter(lambda x: x.startswith('Libs:'), pc)).pop().strip()

        # we only need after '-lmylib' and one entry for library
        d = l.split(lib, 1).pop().split()

        # remove '-l'
        d = map(lambda x: x[2:], d)


        # empty list means no deps
        if d !={}: deps += d

    # Unique list
    result = list(libs)
    map(lambda x: x not in result and result.append(x), deps)
    print(result)
    return result

libinc = [ path_join(ffmpegpath, 'lib') ]

libs = [ 'avformat', 'avcodec', 'avutil', 'swscale','avdevice' ]

if sys.platform in [ 'win32', 'win64' ] :

    libs = static_resolver(libs)
    libinc += [ r'e:\wingmg\lib' ] # it seems some people require this
    incdir = path_join(ffmpegpath, 'include') 

else:
    incdir = [ path_join(ffmpegpath, 'include'), "/usr/include" , "./include" ] 

if (with_numpy):
    incdir = incdir.split()+ list(nd.get_numpy_include_dirs())
print(incdir)
print(libinc)
print(libs)
print(extra_compiler_args)
##################################################################################
if with_numpy:
        ext_modules=[ Extension('pyffmpeg', [ 'FFmpegCAPI.pyx' ],
                       include_dirs = incdir,
                       library_dirs = libinc,
                       libraries = libs,
                       extra_compile_args=extra_compiler_args)
                     ]
else:
        ext_modules=[ Extension('pyffmpeg', [ 'FFmpegCAPI.pyx' ],
                       include_dirs = incdir, 
                       library_dirs = libinc,
                       libraries = libs,
                       extra_compile_args=extra_compiler_args)
                    ]

##################################################################################
setup(
    name = 'pyffmpeg',
    cmdclass = {'build_ext': build_ext},
    version = "0.1",
    ext_modules = ext_modules
)