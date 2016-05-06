#include <stdio.h>
#include <malloc.h>
#include <Python.h>


int compute_overlap(PyObject *A)
{
  int *array = NULL;
  int ii, i;
  int nInts = PyList_Size( A );
  array = malloc( nInts * sizeof( *array ) );

  for ( ii = 0; ii < nInts; ii++ )
  {
    PyObject *oo = PyList_GetItem( A, ii );
    if ( PyInt_Check( oo ) )
    {
      array [ ii ] = ( int ) PyInt_AsLong( oo );
    }
  }
  
  int sum = 0;
  for (i = 0; i < nInts; i++)
  {
    sum += array[i];
  }
  printf("(In C) Total is: %d\n", sum);

  return sum;
}
