int multid(int* npyArray2D, int npyLength1D, int npyLength2D)
{
  int i, j;
  int sum = 0;

  for (i=0;i<npyLength1D;i++)
    for (j=0;j<npyLength2D;j++)
      sum += npyArray2D[i*npyLength2D+j];

  return sum;
}
