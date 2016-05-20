void inplace(double *invec1, int n, double *invec2, int m)
{
  int i;

  for(i=0;i<n && i<m;i++)
    invec1[i] = invec1[i] + invec2[i];

}
