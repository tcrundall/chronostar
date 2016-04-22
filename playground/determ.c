#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

int main()
{
  int MAT_DIM = 4;
  int i,j;
  int myArray[16] = {4, 3, 2, 1, 1,10, 3, 4, 5, 3, 2, -4, 4, 8, 7, 9};

  gsl_matrix *m = gsl_matrix_alloc(MAT_DIM,MAT_DIM);
  
  for (i=0; i<MAT_DIM; i++)
    for (j=0; j<MAT_DIM; j++)
      gsl_matrix_set (m, i, j, (double) myArray[i*MAT_DIM + j]);

  for (i=0; i<MAT_DIM; i++)
    for (j=0; j<MAT_DIM; j++)
      printf("m(%d,%d) = %g\n", i, j,
              gsl_matrix_get (m, i, j));

  double det;
  int signum;
  gsl_permutation *p = gsl_permutation_alloc(m->size1);
  
  gsl_linalg_LU_decomp(m, p, &signum);
  det = gsl_linalg_LU_det(m, signum);
  gsl_permutation_free(p);
  gsl_matrix_free(m);

  printf("Determinant: %g\n", det);

  return 0;
}
