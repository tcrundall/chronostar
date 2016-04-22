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
  double start, end, t_time = 0.0;

  printf("N_MATRIX: %d, ITERATIONS: %d\n", N_MATRIX, ITERATIONS);

  /*int myArray2[36] = {4, 3, 2, 1, 1,10,
                     3, 4, 5, 3, 2, -4,
                     4, 8, 7, 9, 3, 2,
                     10, 12, 1,-2,-5, 1,
                     17, 1, 23, 1, 2,1,
                     34, 2, 1, 2, 1, 1};*/

  //int myArray[MAT_DIM*MAT_DIM*N_MATRIX];
  int *myArray = malloc(N_MATRIX*MAT_DIM*MAT_DIM*sizeof(int));

  gsl_permutation *p;
  gsl_matrix *m_list[N_MATRIX];

  for (count=0; count<ITERATIONS; count++)
  {
    for (i=0; i<N_MATRIX; i++)
      myArray[i] = rand()%100;

   
    for (i=0; i<N_MATRIX; i++)
     m_list[i] = gsl_matrix_alloc(MAT_DIM,MAT_DIM);
    
    for (i=0; i<N_MATRIX; i++)
      for (j=0; j<MAT_DIM; j++)
        for (k=0; k<MAT_DIM; k++)
          gsl_matrix_set (m_list[i], j, k, (double)
                       myArray[i*N_MATRIX + j*MAT_DIM + k]);

    /*for (i=0; i<MAT_DIM; i++)
      for (j=0; j<MAT_DIM; j++)
        printf("m(%d,%d) = %g\n", i, j,
                gsl_matrix_get (m_list[0], i, j));*/

    volatile double det;
    int signum;
    p = gsl_permutation_alloc(m_list[0]->size1);

    start = (double)clock() /(double) CLOCKS_PER_SEC;
    //for (i=0; i<N_MATRIX; i++)

    for (i=0; i<N_MATRIX; i++)
    {
      gsl_linalg_LU_decomp(m_list[0], p, &signum);
      det = gsl_linalg_LU_det(m_list[0], signum);
    }

    end = (double)clock() / (double) CLOCKS_PER_SEC;
    t_time += end - start;
  }

  printf("GSL: %d matrices, %d iterations, cpu time: %fs\n",
                    N_MATRIX, ITERATIONS, t_time);

  gsl_permutation_free(p);
  for (i=0; i<N_MATRIX; i++)
    gsl_matrix_free(m_list[i]);

  return 0;
}
