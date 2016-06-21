double flatten(int len1, double* vec1, int len2, double* vec2);

double flatten2(int len1, double* vec1, int len2, double* vec2);

void range(int *rangevec, int n);

void flatten3(int len1, double* vec1, int len2, double* vec2, int *rangevec, int n);

double sum(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D);


void sum2(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D, double *rangevec, int n);

double get_overlap(double* gr_icov, int gr_dim1, int gr_dim2,
                   double* gr_mn, int gr_mn_dim, double gr_icov_det,
                   double* st_icov, int st_dim1, int st_dim2,
                   double* st_mn, int st_mn_dim, double st_icov_det);
