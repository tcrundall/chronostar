%module overlap
%{
  extern double get_det(PyObject *A);
  extern double get_overlap(PyObject *A, PyObject *a, double A_det,
                            PyObject *B, PyObject *b, double B_det);
%}

extern double get_det(PyObject *A);
extern double get_overlap(PyObject *gr_icov, PyObject *gr_mn,
                    double gr_icov__det, PyObject *st_icov, PyObject *st_mn,
                    double st_icov_det);
