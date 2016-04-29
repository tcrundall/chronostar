all: gslDeterm eigDetermBeta gslLinEq

gslDeterm: gslDeterm.c
	gcc gslDeterm.c -lgsl -lgslcblas -o gslDeterm
    
gslLinEq: gslLinEq.c
	gcc gslLinEq.c -lgsl -lgslcblas -o gslLinEq
    
eigDetermBeta: eigDetermBeta.cpp
	gcc eigDetermBeta.cpp -o eigDetermBeta
	
clean:
	rm -f gslLinEq
	rm -f gslDeterm