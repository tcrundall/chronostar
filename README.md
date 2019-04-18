# Chronostar
A stellar orbit traceback code - *work in progress*

In this repository, we will try to prototype some orbital traceback calculations.

## Project Structure
- benchmarks: A collection of scripts comparing the timings of various
implementations of the overlap integral calculation
- build: of no consequence, a byproduct of running setup.py
- chronostar: housing the integral modules of this software
- data: where astrometry data should live
- demos: toy examples deomstrating usages of various methods, not all
mine
- integration_tests: complicated tests that can take upwards of
30 mins to run, testing everythign workign together
- junkpile: where forgotten code goes to die.
- playground: where I work out how to use things
- results: plots and logs of science runs
- scripts: where the science happens
- tests: unittests that cover the low level stuff - should execute within
half a minute

## Installing

### Dependencies
Python libraries (i.e. pip install [library name])
- astropy
- emcee
- galpy==1.1

Other tools:
- mpirun

#### Note
Unfortunately there seems to be a bug in galpy for our purposes which
requires us to use an older version (see unit_tests/test_galpy.py).
This version has a couple of 
incompatibilities (a bugged scipy importer and outdated Warnings function
signature) which we have to edit manually.
If the most recent installation of galpy suits your needs, then simply
reinstall galpy in a virtual environment (instructions below).

In galpy/orbit_src/FullOrbit.py: to accommodate the existance of scipy 1.0
and later, replace line 7

    if int(scipy.__version__.split('.')[1]) < 10: #pragma: no cover

with

    if int(scipy.__version__.split('.')[0]) == 0 and\
        int(scipy.__version__.split('.')[1]) < 10: #pragma: no cover

In galpy/utils/\_\_init\_\_.py: to ensure the _warning signature has
enough arguments, insert two placeholder arguments such that line 15 goes
from:

    ...
    lineno = -1):
    ...
   
to

    ...
    lineno = -1,
    placeholder1 = None,
    placeholder2 = None):
    ...

Chronostar **can** run with latest releases of galpy, but the unit
tests will fail. Feel free to skip this step for now and inspect the
results of unit_tests/test_galpy.py yourself.

## Setup
Now that python dependencies are met, we need to install swig, a means
to wrap c code such that it can be imported and called by python.

Requirements:
- swig
- gsl (e.g. sudo apt install libgsl0-dev)
- pcre

If these are installed, in the main directory (chronostar) run:

	python setup.py build_ext -b .


### Installing swig
#### Linux (Ubuntu)

    sudo apt install libgsl0-dev
    sudo apt install libpcre3
    sudo apt install libpcre3-dev
    sudo apt install swig

#### Mac

    brew install gsl
    brew install swig  # (will auto install dependency: pcre)

#### Manual installation
If installing swig doesn't work, go to
https://sourceforge.net/projects/swig/
and download swig (target language python)
and, after installing pcre and gsl, follow
instructions in 'INSTALL.txt'

e.g.

    cd [source directory]
    ./configure
    make
    make install

# Testing

Next check everything is in working order. 

    cd unit_tests/
    pytest
    
will run the unittests.
Next, try the following:

    cd integration_tests/
    python test_groupfitter.py
   
This will take ~ 20-40 minutes on a typical machine. A single fit usually
requires approximately 3000 steps to converge, and 100 steps takes about a minute.

Next, (assuming you're on a server with many cores) in the same directory try:

    nohup mpirun -np 19 python test_mpirun.py &
    
If your server has 19+ cores, then run as above. If there's less, then reduce
the number accordingly. Increasing the number of threads is not useful as
each thread is associated with an *emcee* walker, and there are only 18 walkers
being used in this run. Depending on the cores available and how busy the server
is, this should take ~5 minutes. Checkout the walker plots in temp_plots/.

## Files
(will include links to required files)

## Running
(will include instructions on how to run)

[//]: # (# Outdated information)


[//]: # (## EXAMPLE RUN Need to update this section)

[//]: # (For synthetic data:)

[//]: # (./gen_synth_data.py)
[//]: # (./generate_default_tb.py data/synth_data_1groups_*)
[//]: # (./fit_synth_1groups.sh )
[//]: # ()
[//]: # (Investigate results by checking most recent log fils:)
[//]: # (ls logs/ -rtal)
[//]: # ()
[//]: # (You can also plot the corner plots and the lnlike plots. After each run the)
[//]: # (suitable command will be displayed. e.g.)
[//]: # (Logs written)
[//]: # (Plot data saved. Go back and plot with:)
[//]: # (./plot_it.py -t 817882 -f 1 -r 0 -p 1000 -l)
[//]: # (   if not on raijin, or:)
[//]: # (./plot_it.py -t 817882 -f 1 -r 0 -p 1000)
[//]: # ()
[//]: # ()
[//]: # (## Example Setup on Raijin)
[//]: # (if running on node, load required modules:)
[//]: # (	can optionally append these commands to .profile file)
[//]: # (	but I don't fully understand behaviour and can lead to)
[//]: # (	conflicts if modules are preloaded elsewhere...)
[//]: # (	)
[//]: # (	module load python/2.7.11)
[//]: # (	module load python/2.7.11-matplotlib)
[//]: # (	module load gsl/1.15)
[//]: # (	)
[//]: # (nly need to install required python packages once (ever)
[//]: # (ut modules must be loaded first, and I believe these packages)
[//]: # (ill be guaranteed to work for only those modules)
[//]: # ()
[//]: # (	pip install --user astropy)
[//]: # (	pip install --user galpy)
[//]: # (	pip install --user emcee)
[//]: # (	pip install --user corner)
[//]: # (	pip install --user mpi4py)
[//]: # (	pip install --user sympy)
[//]: # ()
[//]: # (build)
[//]: # ()
[//]: # (	python setup.py build_ext -b .)

