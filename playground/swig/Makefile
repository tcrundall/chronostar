all: test

test: test.i
	swig -python test.i
	gcc -c test.c test_wrap.c -I /usr/include/python2.7
	ld -shared test.o test_wrap.o -o _test.so


clean:
	rm -f test.o
	rm -f test_wrap.*
	rm -f _test.so
	rm -f test.pyc
	rm -f test.py
