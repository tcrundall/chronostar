#include <stdio.h>
#include <gsl/gsl_blas.h>

int main()
{
  double a[] = { 0.11, 0.12, 0.13,
                0.21, 0.22, 0.23 };

  double b[] = { 1011, 1012,
                 1021, 1022,
                 1031, 1032 };

  double c[] = { 0.00, 0.00,
                 0.00, 0.00 };

  gsl_matrix_view A = gsl_matrix_view_array(a, 2, 3);
  gsl_matrix_view B = gsl_matrix_view_array(b, 3, 2);
  gsl_matrix_view C = gsl_matrix_view_array(c, 2, 2);

  gsl_matrix *D = gsl_matrix_alloc(2, 3);
  int i, j;
  for (i=1; i<3; i++)
    for (j=1; j<4; j++)
      gsl_matrix_set (D, i-1, j-1, ((float) i)/10 + ((float) j)/100);
  
  for (i=0; i<2; i++)
    for (j=0; j<3; j++)
      printf("D(%d,%d) = %f\n", i, j, gsl_matrix_get(D, i, j));
  
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans,
                 1.0, D, &B.matrix,
                 0.0, &C.matrix);

  printf ("[ %g, %g\n", c[0], c[1]);
  printf ("  %g, %g ]\n", c[2], c[3]);

  return 0;
}
