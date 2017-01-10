int sum(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D);
double get_det(PyObject *A);
double get_overlap2(PyObject *gr_icov, PyObject *gr_mn, double gr_icov_det,
                    PyObject *st_icov, PyObject *st_mn, double st_icov_det);

double get_overlap(double* gr_icov, int gr_dim1, int gr_dim2,
                   double* gr_mn, int gr_mn_dim, double gr_icov_det,
                   double* st_icov, int st_dim1, int st_dim2,
                   double* st_mn, int st_mn_dim, double st_icov_det);

//void get_overlaps(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D, double *rangevec, int n);

void get_overlaps(double* gr_icov, int gr_dim1, int gr_dim2,
                  double* gr_mn, int gr_mn_dim,
                  double gr_icov_det,
                  double* st_icovs, int st_dim1, int st_dim2, int st_dim3,
                  double* st_mns, int st_mn_dim1, int st_mn_dim2,
                  double* st_icov_dets, int st_icov_dets_dim,
                  double* rangevec, int n);
