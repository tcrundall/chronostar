%module overlap

%{
  #define SWIG_FILE_WITH_INIT
  #include "overlap.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) \
      {(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)}

%include "overlap.h"

%clear (int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D);
