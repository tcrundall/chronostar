all: test overlap

test: test.i
	swig -python test.i
	gcc -fPIC -c test.c test_wrap.c -I /pkg/linux/anaconda/include/python2.7
	ld -shared test.o test_wrap.o -o _test.so

overlap: overlap.i
	swig -python overlap.i
	gcc -O3 -fPIC -c overlap.c overlap_wrap.c -I /pkg/linux/anaconda/include/python2.7 -lgsl -lgslcblas
	ld -shared overlap.o overlap_wrap.o -o _overlap.so -lgsl -lgslcblas


clean:
	rm -f test.o
	rm -f test_wrap.*
	rm -f _test.so
	rm -f test.pyc
	rm -f test.py
	rm -f overlap.o
	rm -f overlap_wrap.*
	rm -f _overlap.so
	rm -f overlap.pyc
	rm -f overlap.py
