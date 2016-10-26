# chronostar
A stellar orbit traceback code

In this repository, we will try to prototype some orbital traceback calculations.

For a local build, do:

python setup.py build_ext -b .

Building on Raijin:
	-- everytime
	module load python/2.7.11
	module load python/2.7.11-matplotlib
	module load gsl/1.15
	
	-- just once but after modules are loaded
	pip install --user astropy
	pip install --user galpy
	pip install --user emcee

	-- in chronostar/chronostar
	make
	python test_swig_module.py
