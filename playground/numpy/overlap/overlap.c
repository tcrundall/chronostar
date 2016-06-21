#include <stdio.h>
#include "overlap.h"

double get_overlap(double* gr_icov, int gr_dim1, int gr_dim2,
                   double* gr_mn, int gr_mn_dim, double gr_icov_det,
                   double* st_icov, int st_dim1, int st_dim2,
                   double* st_mn, int st_mn_dim, double st_icov_det)
{
  return gr_icov[0];
}

double sum(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)
{
  int i, j, k;
  double result = 0;

  for (i=0; i<npyLength1D; i++)
    for (j=0; j<npyLength2D; j++)
      for (k=0; k<npyLength3D; k++)
        result += npyArray3D[i*npyLength3D*npyLength2D + k*npyLength2D + j];

  return result;
}

void sum2(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D, double *rangevec, int n)
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


double flatten(int len1, double* vec1, int len2, double* vec2)
{
  int i;
  double d;

  d = 0;
  for (i=0; i<len1; i++)
    d += vec1[i]*vec2[i];

  return d;
}


double flatten2(int len1, double* vec1, int len2, double* vec2)
{
  int i;
  double d;

  d = 0;
  for (i=0; i<len1; i++)
    d += vec1[i]*vec2[i];

  return d;
}

void range(int *rangevec, int n)
{
  int i;

  for (i=0; i<n; i++)
    rangevec[i] = i;
}

void flatten3(int len1, double* vec1, int len2, double* vec2, int *rangevec, int n)
{
  int i;
  double d;

  d = 0;
  for (i=0; i<len1; i++)
    d += vec1[i]*vec2[i];

  for (i=0; i<n; i++)
  {
    rangevec[i] = d;
  }
}
