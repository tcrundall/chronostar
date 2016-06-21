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
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* rangevec, int n)}
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) \
      {(double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) \
      {(double* gr_icov, int gr_dim1, int gr_dim2),
       (double* st_icov, int st_dim1, int st_dim2)}

%apply (double* IN_ARRAY1, int DIM1) \
      {(double* gr_mn, int gr_mn_dim),
       (double* st_mn, int st_mn_dim)}

%include "overlap.h"

%clear (double* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D);
