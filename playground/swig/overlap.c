#include <stdio.h>
#include <malloc.h>
#include <Python.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <string.h>


int get_det(PyObject *A)
{
  int MAT_DIM = 3;
  int *array = NULL;
  int i, j, signum, det;
  int nInts = PyList_Size(A);

  gsl_matrix *m = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_permutation *p;

  p = gsl_permutation_alloc(m->size1);

  for (i=0; i<nInts; i++)
  {
    PyObject *oo = PyList_GetItem(A, i);
    gsl_matrix_set (m, i%MAT_DIM, i/MAT_DIM, (int) PyInt_AsLong(oo));
  }

  gsl_linalg_LU_decomp(m, p, &signum);
  det = gsl_linalg_LU_det(m, signum);

  return det;
}
