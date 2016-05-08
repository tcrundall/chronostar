all: gslDeterm eigDeterm eigDetermBeta gslLinEq gslProduct

gslDeterm: gslDeterm.c
	gcc gslDeterm.c -lgsl -lgslcblas -o gslDeterm
    
gslLinEq: gslLinEq.c
	gcc gslLinEq.c -lgsl -lgslcblas -o gslLinEq

gslProduct: gslProduct.c
	gcc gslProduct.c -lgsl -lgslcblas -o gslProduct
    
eigDetermBeta: eigDetermBeta.cpp
	g++ eigDetermBeta.cpp -o eigDetermBeta -I /usr/local/bin/
	
eigDeterm: eigDeterm.cpp
	g++ -o eigDeterm eigDeterm.cpp -I /usr/local/bin/
	
clean:
	rm -f gslLinEq
	rm -f gslDeterm
	rm -f eigDeterm
	rm -f eigDetermBeta
	rm -f gslProduct
