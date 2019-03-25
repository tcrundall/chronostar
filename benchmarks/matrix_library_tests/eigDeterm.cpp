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


//  MatrixXd *m_list(6,6);

  MatrixXd m = MatrixXd::Random(MAT_DIM, MAT_DIM);
  
  cout << "Here is the matrix m:" << endl << m << endl;
  double determinant;
  cout << "Determinant is: " << m.determinant() << endl;

  return 0;
}
