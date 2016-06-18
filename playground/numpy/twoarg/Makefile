all:
	swig -python flatten.i
	i686-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC -I/usr/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c flatten_wrap.c -o flatten_wrap.o -lgsl -lgslcblas -lm
	i686-linux-gnu-gcc -pthread -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fPIC -I/usr/lib/python2.7/dist-packages/numpy/core/include -I/usr/include/python2.7 -c flatten.c -o flatten.o -lgsl -lgslcblas -lm
	i686-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security flatten_wrap.o flatten.o -o _flatten.so -lgsl -lgslcblas -lm
	
clean:
	rm *.o _* *.so *~ *wrap* *.pyc flatten.py
