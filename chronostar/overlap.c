#include <stdio.h>
#include <malloc.h>
#include <Python.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <string.h>
#include <math.h>

/*
void inplace(double* npyArray3D, int npyLength1D, int npylength2D,
              int npylength2D, double* invec, int n)
{
  int i;
  for (i=0; i<n; i++) {
    invec[i] = npyArray3D[i];
  }
}
*/

void get_overlaps(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D, double *rangevec, int n)
{
  int i, j, k;
  double result = 0;

  for (i=0; i<npyLength1D; i++)
    for (j=0; j<npyLength2D; j++)
      for (k=0; k<npyLength3D; k++)
        result += npyArray3D[i*npyLength3D*npyLength2D + k*npyLength2D + j];

  for (i=0; i<n; i++)
    rangevec[i] = result;
}

int sum(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)
{
  int i, j, k;
  int sum = 0;

  for (i=0; i<npyLength1D; i++)
    for (j=0; j<npyLength2D; j++)
      for (k=0; k<npyLength3D; k++)
        sum += npyArray3D[i*npyLength3D*npyLength2D + k*npyLength2D + j];

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
  double ApB_det, d_temp, result;
  PyObject *o1, *o2;
  gsl_permutation *p;

  gsl_matrix *A        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *B        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *ApB      = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_vector *a        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *b        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *AapBb    = gsl_vector_alloc(MAT_DIM);
  gsl_vector *c        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp2  = gsl_vector_alloc(MAT_DIM);
  gsl_vector *amc      = gsl_vector_alloc(MAT_DIM); //will hold a - c
  gsl_vector *bmc      = gsl_vector_alloc(MAT_DIM); //will hold b - c

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

  // Getting determinant of ApB
  gsl_linalg_LU_decomp(ApB, p, &signum);
  //ApB_det = gsl_linalg_LU_det(ApB, signum);
  ApB_det = fabs(gsl_linalg_LU_det(ApB, signum)); //temp doctoring determinant

  // Solve for c
  gsl_linalg_LU_solve(ApB, p, AapBb, c);

  // Compute the overlap formula
  gsl_vector_set_zero(v_temp);
  gsl_blas_dcopy(a, v_temp);       //v_temp holds a
  gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds a - c
  gsl_blas_dcopy(v_temp, amc);     //amc holds a - c

  // CAN'T HAVE v_temp and v_temp2 be the same vector.
  // Results in 0's being stored in v_temp2.
  gsl_blas_dgemv(CblasNoTrans, 1.0, A, v_temp, 0.0, v_temp2);
  //v_temp2 holds A (a-c)

  result = 0.0;
  gsl_blas_ddot(v_temp2, amc, &d_temp); //d_temp holds (a-c)^T A (a-c)

  result += d_temp;
  
  gsl_vector_set_zero(v_temp);
  gsl_blas_dcopy(b, v_temp);       //v_temp holds b
  gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds b - c
  gsl_blas_dcopy(v_temp, bmc);     //bmc holds b - c

  // CAN'T HAVE v_temp and v_temp2 be the same vector.
  // Results in 0's being stored in v_temp2.
  gsl_blas_dgemv(CblasNoTrans, 1.0, B, v_temp, 0.0, v_temp2);
  //v_temp2 holds A (b-c)

  gsl_blas_ddot(v_temp2, bmc, &d_temp); //d_temp holds (b-c)^T B (b-c)
  result += d_temp;
 
  result = -0.5 * result;
  result = exp(result);

  result *= sqrt((gr_icov_det * st_icov_det/ApB_det)
                / pow(2*M_PI, MAT_DIM));


  // Freeing memory
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_matrix_free(ApB);
  gsl_vector_free(a);
  gsl_vector_free(b);
  gsl_vector_free(AapBb);
  gsl_vector_free(c);
  gsl_vector_free(v_temp);
  gsl_vector_free(v_temp2);
  gsl_vector_free(amc);
  gsl_vector_free(bmc);

  gsl_permutation_free(p);

  return result;
}
