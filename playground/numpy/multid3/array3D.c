#include <math.h>
#include <stdio.h>
int sum(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)
{
  int i, j, k;
  int sum = 0;
  double m = sqrt(9.0);

  for (i=0;i<npyLength1D;i++)
    for (j=0;j<npyLength2D;j++)
      for (k=0;k<npyLength3D;k++)
        sum += npyArray3D[i*npyLength3D*npyLength2D + k*npyLength2D + j];

  printf("%f\n",m);
  return sum;

}
