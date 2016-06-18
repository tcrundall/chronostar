%module flatten

%{
  #define SWIG_FILE_WITH_INIT
  #include "flatten.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {(int len1, double* vec1), (int len2, double* vec2)}
%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* rangevec, int n)}

%include "flatten.h"
