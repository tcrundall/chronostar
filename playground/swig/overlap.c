#include <stdio.h>
#include <malloc.h>
#include <Python.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <string.h>

double get_det(PyObject *A)
{
  int MAT_DIM = 6;
  int *array = NULL;
  int i, j, signum;
  double det;
  int nInts = PyList_Size(A);

  gsl_matrix *m = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_permutation *p;

  p = gsl_permutation_alloc(m->size1);
  
  for (i=0; i<nInts; i++)
  {
    PyObject *oo = PyList_GetItem(A, i);
    //printf("%6.2f\n",PyFloat_AS_DOUBLE(oo));
    gsl_matrix_set (m, i%MAT_DIM, i/MAT_DIM, PyFloat_AS_DOUBLE(oo));
  }
  
  gsl_linalg_LU_decomp(m, p, &signum);
  det = gsl_linalg_LU_det(m, signum);
  //printf("%6.2f\n",det);
  return det;
}

double get_overlap(PyObject *gr_icov, PyObject *gr_mn, double gr_icov_det,
                   PyObject *st_icov, PyObject *st_mn, double st_icov_det)
{
  int MAT_DIM = 6;
  int i, j, signum;
  double ApB_det;
  PyObject *o1, *o2;
  gsl_permutation *p;

  gsl_matrix *A     = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *B     = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *ApB   = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_vector *a     = gsl_vector_alloc(MAT_DIM);
  gsl_vector *b     = gsl_vector_alloc(MAT_DIM);
  gsl_vector *AapBb = gsl_vector_alloc(MAT_DIM);
  gsl_vector *c     = gsl_vector_alloc(MAT_DIM);

  p = gsl_permutation_alloc(A->size1);

  //Inserting values into matricies and vectors
  for (i=0; i<MAT_DIM; i++)
  {
    for (j=0; j<MAT_DIM; j++)
    {
      o1 = PyList_GetItem(gr_icov, i*MAT_DIM + j);
      gsl_matrix_set (A, i, j, PyFloat_AS_DOUBLE(o1));
      o2 = PyList_GetItem(st_icov, i*MAT_DIM + j);
      gsl_matrix_set (B, i, j, PyFloat_AS_DOUBLE(o2));
    }
  }

  for (i=0; i<MAT_DIM; i++)
  {
    o1 = PyList_GetItem(gr_mn, i);
    gsl_vector_set (a, i, PyFloat_AS_DOUBLE(o1));
    o2 = PyList_GetItem(st_mn, i);
    gsl_vector_set (b, i, PyFloat_AS_DOUBLE(o2));
  }

  // Adding A and B together and storing in ApB
  gsl_matrix_set_zero(ApB);
  gsl_matrix_add(ApB, A);
  gsl_matrix_add(ApB, B);

  // Storing the result A*a + B*b in AapBb
  gsl_vector_set_zero(AapBb);
  gsl_blas_dgemv(CblasNoTrans,
                 1.0, A, a,
                 1.0, AapBb);
  gsl_blas_dgemv(CblasNoTrans,
                 1.0, B, b,
                 1.0, AapBb);

  for (i=0; i<MAT_DIM; i++)
  {
    printf("AapBb(%d): %f\n", i, gsl_vector_get(AapBb, i));
  }

  // Getting determinant of ApB
  gsl_linalg_LU_decomp(ApB, p, &signum);
  ApB_det = gsl_linalg_LU_det(ApB, signum);

  printf("Determinant of ApB is: %f\n", ApB_det);

  // Solve for c
  gsl_linalg_LU_solve(ApB, p, AapBb, c);


  for (i=0; i<MAT_DIM; i++)
  {
    printf("C(%d): %f\n", i, gsl_vector_get(c, i));
  }

  //printing for sanity reasons
  /*
  for (i=0; i<MAT_DIM; i++)
    for (j=0; j<MAT_DIM; j++)
      printf ("A(%d, %d) = %g\n", i, j, gsl_matrix_get (A, i, j));
  

  for (i=0; i<MAT_DIM; i++)
    for (j=0; j<MAT_DIM; j++)
      printf ("ApB(%d, %d) = %g\n", i, j, gsl_matrix_get (ApB, i, j));
  */

  // Freeing memory
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_matrix_free(ApB);
  gsl_vector_free(a);
  gsl_vector_free(b);

  return 0.0;
}
