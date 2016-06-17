%module overlap
%{
  #define SWIG_FILE_WITH_INIT
  extern double get_det(PyObject *A);
  extern double get_overlap(PyObject *A, PyObject *a, double A_det,
                            PyObject *B, PyObject *b, double B_det);
  #include "overlap.h"
%}

%include "numpy.i"
%init %{
  import_array();
%}

%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) \
       {(int* npyArray3D, int npyLenght1D, int npyLength2D, int npyLength3D)}

extern double get_det(PyObject *A);
extern double get_overlap(PyObject *gr_icov, PyObject *gr_mn,
                    double gr_icov__det, PyObject *st_icov, PyObject *st_mn,
                    double st_icov_det);
%include "overlap.h"

%clear (int* npyArray3D, int npLength1D, int npyLength2D, int npyLength3D);
