#include <stdio.h>
#include "flatten.h"

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
