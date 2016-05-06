all: gslDeterm eigDeterm eigDetermBeta gslLinEq

gslDeterm: gslDeterm.c
	gcc gslDeterm.c -lgsl -lgslcblas -o gslDeterm
    
gslLinEq: gslLinEq.c
	gcc gslLinEq.c -lgsl -lgslcblas -o gslLinEq
    
eigDetermBeta: eigDetermBeta.cpp
	g++ eigDetermBeta.cpp -o eigDetermBeta -I /usr/local/bin/
	
eigDeterm: eigDeterm.cpp
	g++ -o eigDeterm eigDeterm.cpp -I /usr/local/bin/
	
clean:
	rm -f gslLinEq
	rm -f gslDeterm
	rm -f eigDeterm
	rm -f eigDetermBeta
