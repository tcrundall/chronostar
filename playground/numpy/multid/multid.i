%module multid

%{
  #define SWIG_FILE_WITH_INIT
  #include "multid.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) \
      {(int* npyArray2D, int npyLength1D, int npyLength2D)}

%include "multid.h"

%clear (int* npyArray2D, int npyLength1D, int npyLength2D);
