#include <math.h>
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <Python.h>
int sum(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)
{
  int i, j, k;
  int sum = 0;
  double dummy = sqrt(9.0);
  int MAT_DIM = 6;

  gsl_matrix *m = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  for (i=0; i < MAT_DIM*MAT_DIM; i++) {
    gsl_matrix_set (m, i%MAT_DIM, i/MAT_DIM, (double) npyArray3D[i]);
  }

  for (i=0; i < MAT_DIM*MAT_DIM; i++) {
    printf("%f, ", gsl_matrix_get (m, i%MAT_DIM, i/MAT_DIM));
  }
  printf("\n");

  for (i=0;i<npyLength1D;i++)
    for (j=0;j<npyLength2D;j++)
      for (k=0;k<npyLength3D;k++)
        sum += npyArray3D[i*npyLength3D*npyLength2D + k*npyLength2D + j];

  printf("%f\n",dummy);
  return sum;
}

double get_det(PyObject *A)
{
  int MAT_DIM = 6;
  int i, signum;
  double det;
  int nInts = PyList_Size(A);

  gsl_matrix *m = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_permutation *p;

  p = gsl_permutation_alloc(m->size1);

  for (i=0; i<nInts; i++)
  {
    PyObject *oo = PyList_GetItem(A, i);
    gsl_matrix_set (m, i%MAT_DIM, i/MAT_DIM, PyFloat_AS_DOUBLE(oo));
  }

  gsl_linalg_LU_decomp(m, p, &signum);
  det = gsl_linalg_LU_det(m, signum);

  return det;
}
