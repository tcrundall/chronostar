%module array3D

%{
  #define SWIG_FILE_WITH_INIT
  #include "array3D.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) \
      {(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)}

%include "array3D.h"

%clear (int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D);
