
all: optix

optix: PyOptiXModule.c PyOptiXDecls.h PyOptiXUtil.h setup.py
	python3 setup.py build
	sudo python3 setup.py install 

PyOptiXModule%c PyOptiXDecls%h: create_python_bindings.py
	#python3 create_python_bindings.py /Users/keith/Code/rtsdk/rel3.x/include/ 
	#python3 create_python_bindings.py /Users/kmorley/Code/rtsdk/rtmain/include/ 
	python3 create_python_bindings.py /home/kmorley/Code/rtsdk/rtmain/include/ 

clean:
	rm -rf build
