
all: optix

optixmodule.c: create_python_bindings.py
	./create_python_bindings.py /Users/keith/Code/rtsdk/rtmain/include/ > optixmodule.c

optix: optixmodule.c setup.py
	python setup.py build
	sudo python setup.py install 

