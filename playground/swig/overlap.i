%module overlap
%{
  extern int compute_overlap(PyObject *A);
%}

extern int compute_overlap(PyObject *A);
