%module overlap
%{
  extern double get_det(PyObject *A);
  extern double get_overlap(PyObject *A, PyObject *a,
                            PyObject *B, PyObject *b, double bg_dens);
%}

extern double get_det(PyObject *A);
extern double get_overlap(PyObject *A, PyObject *a,
                          PyObject *B, PyObject *b, double bg_dens);
