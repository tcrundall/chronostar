# chronostar
A stellar orbit traceback code

In this repository, we will try to prototype some orbital traceback calculations.

For a local build, do:

	python setup.py build_ext -b .

Building on Raijin:
	-- if running in node, load required modules:
	--    (can optionally append these commands to .profile file
	--     but I don't fully understand behaviour and can lead to
	--     conflicts if modules are preloaded elsewhere...)
	module load python/2.7.11
	module load python/2.7.11-matplotlib
	module load gsl/1.15
	
	-- only need to install required python packages once (ever)
	-- but modules must be loaded first, and I believe these packages
	-- will be guaranteed to work for only those modules
	pip install --user astropy
	pip install --user galpy
	pip install --user emcee
	pip install --user corner
	pip install --user mpi4py
	pip install --user sympy

	-- build
	python setup.py build_ext -b .

	-- to test
	python test_swig_module.py

--------------------------------------------
--                  FILES                 --
--------------------------------------------

