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

--------------------------------------------
--              EXAMPLE RUN               --
--------------------------------------------
For synthetic data:
./gen_synth_data.py
./generate_default_tb.py data/synth_data_1groups_*
./fit_synth_1groups.sh data/synth_data_1groups_*

Investigate results by checking most recent log fils:
ls logs/ -rtal

You can also plot the corner plots and the lnlike plots. After each run the
suitable command will be displayed. e.g.
Logs written
Plot data saved. Go back and plot with:
./plot_it.py -t 817882 -f 1 -r 0 -p 1000 -l
   if not on raijin, or:
./plot_it.py -t 817882 -f 1 -r 0 -p 1000
