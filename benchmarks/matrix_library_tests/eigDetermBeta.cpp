#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>

/*
 * A benchmark to test performance of calculating determinants of 
 * 6x6 matrices with the Eigen linear algebra package.
 */

using namespace std;
using namespace Eigen;

int main(int argc, char * argv[])
{
  if (argc != 3 || strcmp(argv[1], "-h") == 0)
  {
    cout << "Usage: ./eigDeterm N_MATRIX(<=1000) ITERATIONS(~1000)"
      << endl;
    return 1;
  }

  srand(time(NULL));
  int MAT_DIM = 6;
  int N_MATRIX = atoi(argv[1]);
  int ITERATIONS = atoi(argv[2]);
  int i, j, k, count;
  double start, end, t_time = 0.0;

  printf("N_MATRIX: %d, ITERATIONS: %d\n", N_MATRIX, ITERATIONS);

  //Setting up array of matrices
  MatrixXd *m_list = new MatrixXd[N_MATRIX];

  //Initialise each matrix in m_list as random 6x6 matrices
  for (i=0; i<N_MATRIX; i++)
  {
    m_list[i] = MatrixXd::Random(MAT_DIM, MAT_DIM);
  }

  start = (double)clock() / (double) CLOCKS_PER_SEC;
  
  //Calculating determinants of matrices
  for(i=0; i<ITERATIONS; i++)
  {
    for(j=0; j<N_MATRIX; j++)
    {
    volatile double determinant;
    determinant = m_list[j].determinant();
    }
  }
  end = (double)clock() / (double) CLOCKS_PER_SEC;

  cout << "Total CPU time: " << end - start << endl;

  return 0;
}
