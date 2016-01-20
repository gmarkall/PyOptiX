
all: optix

optix: PyOptiXModule.c PyOptiXDecls.h PyOptiXUtil.h setup.py
	python setup.py build
	sudo python setup.py install 

PyOptiXModule%c PyOptiXDecls%h: create_python_bindings.py
	python create_python_bindings.py /Users/keith/Code/rtsdk/rel3.x/include/ 
	#python create_python_bindings.py /Users/kmorley/Code/rtsdk/rtmain/include/ 

clean:
	rm -rf build
