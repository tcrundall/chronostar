#include <iostream>
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

  MatrixXd m = MatrixXd::Random(6, 6);
  cout << "Here is the matrix m:" << endl << m << endl;
  /*Matrix3d inverse;
  bool invertible;
  double determinant;
  m.computeInverseAndDetWithCheck(inverse,determinant, invertible);
  cout << "Its determinant is " << determinant << endl;
  if (invertible) {
    cout << "It is invertible, and its inverse is:" << endl << inverse << endl;
  }
  else {
    cout << "It is not invertible." << endl;
  }
*/
  return 0;
}
