import os
import subprocess

def test_mpi():
    bash_command = 'mpirun -np 2 python mpi_script.py'
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    assert error is None

