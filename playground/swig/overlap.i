%module overlap
%{
  extern double get_det(PyObject *A);
%}

extern double get_det(PyObject *A);
