%module inplace

%{
  #define SWIG_FILE_WITH_INIT
  #include "inplace.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double* invec, int n)}

%include "inplace.h"
