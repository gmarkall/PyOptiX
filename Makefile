
all: optix

optix: PyOptiXModule.c PyOptiXDecls.h PyOptiXUtil.h setup.py
	python setup.py build
	sudo python setup.py install 

PyOptiXModule%c PyOptiXDecls%h: create_python_bindings.py
	#python create_python_bindings.py /Users/keith/Code/rtsdk/rtmain/include/ 
	python create_python_bindings.py /Users/kmorley/Code/rtsdk/rtmain/include/ 

