#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <time.h>
#include <string.h>

int main(int argc, char * argv[])
{
  if (argc != 3 || strcmp(argv[1], "-h") == 0)
  {
    printf("Usage: ./gslDeterm N_MATRIX(<=1000) ITERATIONS(~1000)\n");
    return 1;
  }
    
  srand(time(NULL));
  int MAT_DIM = 6;
  int N_MATRIX = atoi(argv[1]);
  int ITERATIONS = atoi(argv[2]);
  int i,j,k, count;
  double start, end, t_time = 0.0, dummy;

  printf("N_MATRIX: %d, ITERATIONS: %d\n", N_MATRIX, ITERATIONS);

  gsl_permutation *p;
  gsl_matrix *m_list[N_MATRIX];
  gsl_vector *x = gsl_vector_alloc (MAT_DIM);
  int signum;

  double b_data[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  gsl_vector_view b = gsl_vector_view_array (b_data, 6);

  // Do this ITERATIONS times, timing only the loop with the determinant
  // finding
  for (count=0; count<ITERATIONS; count++)
  {
    // Initialise a list of gsl arrays
    for (i=0; i<N_MATRIX; i++)
     m_list[i] = gsl_matrix_alloc(MAT_DIM,MAT_DIM);
    
    // Initialise values to random integers 0 -> 99
    for (i=0; i<N_MATRIX; i++)
      for (j=0; j<MAT_DIM; j++)
        for (k=0; k<MAT_DIM; k++)
          gsl_matrix_set (m_list[i], j, k, rand()%100);

    p = gsl_permutation_alloc(m_list[0]->size1);

    //start time: find the determinant of each matrix in
    //m_list
    start = (double)clock() /(double) CLOCKS_PER_SEC;

    for (i=0; i<N_MATRIX; i++)
    {
      gsl_linalg_LU_decomp(m_list[i], p, &signum);
      gsl_linalg_LU_solve(m_list[i], p, &b.vector, x);
    }

    end = (double)clock() / (double) CLOCKS_PER_SEC;
    t_time += end - start;
  }

  printf("GSL: %d matrices, %d iterations, cpu time: %fs\n",
                    N_MATRIX, ITERATIONS, t_time);

  gsl_permutation_free(p);
  for (i=0; i<N_MATRIX; i++)
    gsl_matrix_free(m_list[i]);
  gsl_vector_free (x);

  return 0;
}
