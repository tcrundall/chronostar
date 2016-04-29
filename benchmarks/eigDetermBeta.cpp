#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Core>

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

  //Setting up matrix list
  MatrixXd *m_list = new MatrixXd[N_MATRIX];
  for (i=0; i<N_MATRIX; i++)
  {
    m_list[i] = MatrixXd::Random(MAT_DIM, MAT_DIM);
  }

  start = (double)clock() / (double) CLOCKS_PER_SEC;
  
  for(i=0; i<ITERATIONS; i++)
  {
    for(j=0; j<N_MATRIX; j++)
    {
    //cout << "Here is the matrix m:" << endl << m_list[j] << endl;
    volatile double determinant;
    determinant = m_list[j].determinant();
    //cout << "Determinant is: " << m_list[j].determinant() << endl;
    }
  }
  end = (double)clock() / (double) CLOCKS_PER_SEC;

  cout << "Total CPU time: " << end - start << endl;

  return 0;
}
