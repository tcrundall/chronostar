#include <stdio.h>
//#include <malloc.h>
#include <Python.h>

void addInt(int x, int y)
{
  printf("%d\n", x+y);
}

void sumList(PyObject *int_list)
{
  int *array = NULL;
  int ii, i;
  if ( PyList_Check( int_list ) )
  {
    int nInts = PyList_Size( int_list );
    array = malloc( nInts * sizeof( *array ) );
    for ( ii = 0; ii < nInts; ii++ )
    {
      PyObject *oo = PyList_GetItem( int_list, ii );
      if ( PyInt_Check( oo ) )
      {
        array [ ii ] = ( int ) PyInt_AsLong( oo );
      }
    }
    
    int sum = 0;
    for (i = 0; i < nInts; i++)
    {
      printf("%d, ", array[i]);
      sum += array[i];
    }
    printf("\nTotal is: %d\n", sum);
  }
}
