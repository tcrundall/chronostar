int sum(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D);
double get_det(PyObject *A);
double get_overlap(PyObject *gr_icov, PyObject *gr_mn, double gr_icov__det,
                   PyObject *st_icov, PyObject *st_mn, double st_icov_det);
