
all: optix

optix: optixmodule.c PyOptiXUtil.h setup.py
	/usr/bin/python setup.py build
	sudo /usr/bin/python setup.py install 

optixmodule.c: create_python_bindings.py
	/usr/bin/python create_python_bindings.py /Users/keith/Code/rtsdk/rtmain/include/ > optixmodule.c
	#/usr/bin/python create_python_bindings.py /Users/kmorley/Code/rtsdk/rtmain/include/ > optixmodule.c


