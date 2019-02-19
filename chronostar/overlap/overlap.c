#include <stdio.h>

//#ifndef DARWIN 
//#include <malloc.h>
//#endif

#include <stddef.h>
#include <Python.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <string.h>
#include <math.h>

/*
 *  A simple helper function to display the contents of a 2-D gsl matrix
 */
int print_matrix(FILE *f, const gsl_matrix *m)
{
  int status, n = 0;
  size_t i, j;

  for (i = 0; i < m->size1; i++) {
    for (j = 0; j < m->size2; j++) {
      if ((status = fprintf(f,"%g ", gsl_matrix_get(m, i, j))) < 0)
        return -1;
      n += status;
    }

    if ((status = fprintf(f, "\n")) < 0)
      return -1;
    n += status;
  }
  return n;
}

/*
 *  A simple helper function to display the contennts of a 1-D gsl vector
 */
int print_vector(FILE *f, const gsl_vector *m)
{
  int status, n = 0;
  size_t i;

  for (i = 0; i < m->size; i++) {
    if ((status = fprintf(f, "%g ", gsl_vector_get(m, i))) < 0)
      return -1;
    n += status;
  }
  return n;
}


void print_mat(double* mat, int dim1, int dim2)
{
  int i, j;
  printf("\n");
  for (i=0; i<dim1; i++) {
    for (j=0; j<dim2; j++) {
      printf("%6.2f,", mat[i*dim1 + j]);
    }
    printf("\n");
  }
}
void print_vec(double* vec, int dim1)
{
  int i;
  printf("\n");
  for (i=0; i<dim1; i++) {
    printf("%6.2f,", vec[i]);
  }
  printf("\n");
}

/* Function: get_lnoverlaps
 * ------------------------
 *   Calculates the log overlap (convolution) with a set of 6D Gaussians with
 *   a single 6D Gaussian, each in turn.
 *
 *   In Chronostar, this is used to see how well the kinematic properties
 *   of stars overlap with a proposed Gaussian distribution.
 *
 * Paramaters
 *  name      type            description
 * ----------
 *  gr_cov        (6*6 npArray)   group's covariance matrix
 *  gr_mn         (1*6 npArray)   group's central estimate (mean)
 *  st_covs       (n*6*6 npArray) array of each star's cov matrix
 *  st_mns:       (n*6 npArray)   array of each star's central estimate
 *  lnols_output: (n npArray)     used to store and return calculated overlaps
 *  nstars:       (int)           number of stars, used for array dimensions
 *
 * Returns
 * -------
 *  (nstars) array of calculated log overlaps of every star with component
 *      thanks to Swig magic the result is stored in `lnols_output` which is
 *      returned as output to the python call as a numpy array
 *
 * Notes
 * -----
 *   For each star calculates:
 *     log(1/sqrt( (2*PI)^6*det(C) ) * exp( -0.5*(b-a)^T*(C)^-1*(b-a) )
 *   where
 *     C = st_cov + gr_cov
 *   and
 *     a = gr_mn, b = st_mn
 *   Expanding and simplifying this becomes:
 *     -0.5[ 6*ln(2*PI) + ln(|C|) + (b-a)^T*(C^-1)*(b-a) ]
 *
 *   Stark improvement on previous implementations. Doesn't require input as
 *   inverse covariance matrices. Never performs a matrix inversion.
 */
void get_lnoverlaps(
  double* gr_cov, int gr_dim1, int gr_dim2,
  double* gr_mn, int gr_mn_dim,
  double* st_covs, int st_dim1, int st_dim2, int st_dim3,
  double* st_mns, int st_mn_dim1, int st_mn_dim2,
  double* lnols_output, int n
  )
{
  // ALLOCATE MEMORY
  int star_count = 0;
  int MAT_DIM = gr_dim1; //Typically set to 6
  int i, j, signum;
  double d_temp, result, ln_det_BpA;
  gsl_permutation *p1;

  gsl_matrix *BpA      = gsl_matrix_alloc(MAT_DIM, MAT_DIM); //will hold (B+A)
  gsl_vector *bma      = gsl_vector_alloc(MAT_DIM);          //will hold b - a
  gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);

  p1 = gsl_permutation_alloc(BpA->size1);

  // Go through each star, calculating and storing overlap
  for (star_count=0; star_count<n; star_count++) {
    // INITIALISE STAR MATRIX
    for (i=0; i<MAT_DIM; i++)
      for (j=0; j<MAT_DIM; j++)
        //performing st_cov+gr_cov as part of the initialisation
        gsl_matrix_set(
          BpA,i,j,
          st_covs[star_count*MAT_DIM*MAT_DIM+i*MAT_DIM+j] +
          gr_cov[i*MAT_DIM+j]
        );

    // INITIALISE CENTRAL ESTIMATES
    // performing st_mn - gr_mn as part of the initialisation
    for (i=0; i<MAT_DIM; i++) {
      gsl_vector_set(
        bma, i,
        st_mns[star_count*MAT_DIM + i] - gr_mn[i]
      );
    }

    // CALCULATE OVERLAPS
    // Performed in 4 stages
    // Calc and sum up the inner terms:
    // 1) 6 ln(2pi)
    // 2) ln(|C|)
    // 3) (b-a)^T(C^-1)(b-a)
    // Then apply -0.5 coefficient

    // 1) Calc 6 ln(2pi)
    result = 6*log(2*M_PI);

    // 2) Get log determiant of C
    gsl_linalg_LU_decomp(BpA, p1, &signum);
    ln_det_BpA = log(fabs(gsl_linalg_LU_det(BpA, signum)));
    result += ln_det_BpA;

    // 3) Calc (b-a)^T(C^-1)(b-a)
    gsl_vector_set_zero(v_temp);
    gsl_linalg_LU_solve(BpA, p1, bma, v_temp); /* v_temp holds (B+A)^-1 (b-a) *
                                                * utilises `p1` as calculated *
                                                * above                       */
    gsl_blas_ddot(v_temp, bma, &d_temp); //d_temp holds (b-a)^T (B+A)-1 (b-a)
    result += d_temp;

    // 4) Apply coefficient
    result *= -0.5;

    // STORE RESULT 'lnols_output'
    lnols_output[star_count] = result;
  }

  // DEALLOCATE THE MEMORY
  gsl_matrix_free(BpA);
  gsl_vector_free(bma);
  gsl_vector_free(v_temp);
  gsl_permutation_free(p1);
}

/* NOTE:
 * Everything below this line is left simply for correctness comparisons
 */
double get_overlap(double* gr_icov, int gr_dim1, int gr_dim2,
                   double* gr_mn, int gr_mn_dim, double gr_icov_det,
                   double* st_icov, int st_dim1, int st_dim2,
                   double* st_mn, int st_mn_dim, double st_icov_det)
{
  int MAT_DIM = gr_dim1;
  int i, j, signum;
  double ApB_det, d_temp, result;
  gsl_permutation *p;

  gsl_matrix *A        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *B        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *ApB      = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_vector *a        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *b        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *AapBb    = gsl_vector_alloc(MAT_DIM);
  gsl_vector *c        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp2  = gsl_vector_alloc(MAT_DIM);
  gsl_vector *amc      = gsl_vector_alloc(MAT_DIM); //will hold a - c
  gsl_vector *bmc      = gsl_vector_alloc(MAT_DIM); //will hold b - c

  p = gsl_permutation_alloc(A->size1);

//for (l=0; l<1; l++){ //DELETE

  //Inserting values into matricies and vectors
  for (i=0; i<MAT_DIM; i++)
  {
    for (j=0; j<MAT_DIM; j++)
    {
      gsl_matrix_set (A, i, j, gr_icov[i*MAT_DIM + j]);
      gsl_matrix_set (B, i, j, st_icov[i*MAT_DIM + j]);
    }
  }

  for (i=0; i<MAT_DIM; i++)
  {
    gsl_vector_set (a, i, gr_mn[i]);
    gsl_vector_set (b, i, st_mn[i]);
  }

  // Adding A and B together and storing in ApB
  gsl_matrix_set_zero(ApB);
  gsl_matrix_add(ApB, A);
  gsl_matrix_add(ApB, B);

  // Storing the result A*a + B*b in AapBb
  gsl_vector_set_zero(AapBb);
  gsl_blas_dsymv(CblasUpper, 1.0, A, a, 1.0, AapBb);
  gsl_blas_dsymv(CblasUpper, 1.0, B, b, 1.0, AapBb);

  // Getting determinant of ApB
  gsl_linalg_LU_decomp(ApB, p, &signum);
  //ApB_det = gsl_linalg_LU_det(ApB, signum);
  ApB_det = fabs(gsl_linalg_LU_det(ApB, signum)); //temp doctoring determinant

  // Solve for c
  gsl_linalg_LU_solve(ApB, p, AapBb, c);

  // Compute the overlap formula
  gsl_vector_set_zero(v_temp);
  gsl_blas_dcopy(a, v_temp);       //v_temp holds a
  gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds a - c
  gsl_blas_dcopy(v_temp, amc);     //amc holds a - c

  // CAN'T HAVE v_temp and v_temp2 be the same vector.
  // Results in 0's being stored in v_temp2.
  gsl_blas_dsymv(CblasUpper, 1.0, A, v_temp, 0.0, v_temp2);
  //v_temp2 holds A (a-c)

  result = 0.0;
  gsl_blas_ddot(v_temp2, amc, &d_temp); //d_temp holds (a-c)^T A (a-c)

  result += d_temp;
  
  gsl_vector_set_zero(v_temp);
  gsl_blas_dcopy(b, v_temp);       //v_temp holds b
  gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds b - c
  gsl_blas_dcopy(v_temp, bmc);     //bmc holds b - c

  // CAN'T HAVE v_temp and v_temp2 be the same vector.
  // Results in 0's being stored in v_temp2.
  gsl_blas_dsymv(CblasUpper, 1.0, B, v_temp, 0.0, v_temp2);
  //v_temp2 holds A (b-c)

  gsl_blas_ddot(v_temp2, bmc, &d_temp); //d_temp holds (b-c)^T B (b-c)
  result += d_temp;
 
  result = -0.5 * result;
  result = exp(result);

  result *= sqrt((gr_icov_det * st_icov_det/ApB_det)
                / pow(2*M_PI, MAT_DIM));
//} //DELETE THIS 

  // Freeing memory
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_matrix_free(ApB);
  gsl_vector_free(a);
  gsl_vector_free(b);
  gsl_vector_free(AapBb);
  gsl_vector_free(c);
  gsl_vector_free(v_temp);
  gsl_vector_free(v_temp2);
  gsl_vector_free(amc);
  gsl_vector_free(bmc);

  gsl_permutation_free(p);

  return result;
}

double get_overlap2(PyObject *gr_icov, PyObject *gr_mn, double gr_icov_det,
                    PyObject *st_icov, PyObject *st_mn, double st_icov_det)
{
  int MAT_DIM = 6;
  int i, j, signum;
  double ApB_det, d_temp, result;
  PyObject *o1, *o2;
  gsl_permutation *p;

  gsl_matrix *A        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *B        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *ApB      = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_vector *a        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *b        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *AapBb    = gsl_vector_alloc(MAT_DIM);
  gsl_vector *c        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp2  = gsl_vector_alloc(MAT_DIM);
  gsl_vector *amc      = gsl_vector_alloc(MAT_DIM); //will hold a - c
  gsl_vector *bmc      = gsl_vector_alloc(MAT_DIM); //will hold b - c

  p = gsl_permutation_alloc(A->size1);

  //Inserting values into matricies and vectors
  for (i=0; i<MAT_DIM; i++)
  {
    for (j=0; j<MAT_DIM; j++)
    {
      o1 = PyList_GetItem(gr_icov, i*MAT_DIM + j);
      gsl_matrix_set (A, i, j, PyFloat_AS_DOUBLE(o1));
      o2 = PyList_GetItem(st_icov, i*MAT_DIM + j);
      gsl_matrix_set (B, i, j, PyFloat_AS_DOUBLE(o2));
    }
  }

  for (i=0; i<MAT_DIM; i++)
  {
    o1 = PyList_GetItem(gr_mn, i);
    gsl_vector_set (a, i, PyFloat_AS_DOUBLE(o1));
    o2 = PyList_GetItem(st_mn, i);
    gsl_vector_set (b, i, PyFloat_AS_DOUBLE(o2));
  }

  // Adding A and B together and storing in ApB
  gsl_matrix_set_zero(ApB);
  gsl_matrix_add(ApB, A);
  gsl_matrix_add(ApB, B);

  // Storing the result A*a + B*b in AapBb
  gsl_vector_set_zero(AapBb);
  gsl_blas_dgemv(CblasNoTrans,
                 1.0, A, a,
                 1.0, AapBb);
  gsl_blas_dgemv(CblasNoTrans,
                 1.0, B, b,
                 1.0, AapBb);

  // Getting determinant of ApB
  gsl_linalg_LU_decomp(ApB, p, &signum);
  //ApB_det = gsl_linalg_LU_det(ApB, signum);
  ApB_det = fabs(gsl_linalg_LU_det(ApB, signum)); //temp doctoring determinant

  // Solve for c
  gsl_linalg_LU_solve(ApB, p, AapBb, c);

  // Compute the overlap formula
  gsl_vector_set_zero(v_temp);
  gsl_blas_dcopy(a, v_temp);       //v_temp holds a
  gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds a - c
  gsl_blas_dcopy(v_temp, amc);     //amc holds a - c

  // CAN'T HAVE v_temp and v_temp2 be the same vector.
  // Results in 0's being stored in v_temp2.
  gsl_blas_dgemv(CblasNoTrans, 1.0, A, v_temp, 0.0, v_temp2);
  //v_temp2 holds A (a-c)

  result = 0.0;
  gsl_blas_ddot(v_temp2, amc, &d_temp); //d_temp holds (a-c)^T A (a-c)

  result += d_temp;
  
  gsl_vector_set_zero(v_temp);
  gsl_blas_dcopy(b, v_temp);       //v_temp holds b
  gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds b - c
  gsl_blas_dcopy(v_temp, bmc);     //bmc holds b - c

  // CAN'T HAVE v_temp and v_temp2 be the same vector.
  // Results in 0's being stored in v_temp2.
  gsl_blas_dgemv(CblasNoTrans, 1.0, B, v_temp, 0.0, v_temp2);
  //v_temp2 holds A (b-c)

  gsl_blas_ddot(v_temp2, bmc, &d_temp); //d_temp holds (b-c)^T B (b-c)
  result += d_temp;
 
  result = -0.5 * result;
  result = exp(result);

  result *= sqrt((gr_icov_det * st_icov_det/ApB_det)
                / pow(2*M_PI, MAT_DIM));


  // Freeing memory
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_matrix_free(ApB);
  gsl_vector_free(a);
  gsl_vector_free(b);
  gsl_vector_free(AapBb);
  gsl_vector_free(c);
  gsl_vector_free(v_temp);
  gsl_vector_free(v_temp2);
  gsl_vector_free(amc);
  gsl_vector_free(bmc);

  gsl_permutation_free(p);

  return result;
}

/* Main function which performs fastest so far:
 * --parameters--
 *  group_icov     (6*6 npyArray) the group's inverse covariance matrix
 *  group_mn       (1*6 npyArray) which is the group's mean kinematic info
 *  group_icov_det (flt)          the determinent of the group_icov
 *  Bs             (nstars*6*6)   an array of each star's icov matrix
 *  bs:            (nstars*6)     an array of each star's mean kinematic info
 *  B_dets:        (nstars)       an array of the determinent of each icov
 *  nstars:        (int)          number of stars, used to determine the size
 *                          of npyArray which will return calculated overlaps)
 *
 * returns: (nstars) array of calculated overlaps of every star with 1 group
 *
 * todo: instead of calling internal function actually use cblas functions
 *          this will save time on the reallocation and deallocation
 */
void get_overlaps(double* gr_icov, int gr_dim1, int gr_dim2,
                  double* gr_mn, int gr_mn_dim,
                  double gr_icov_det,
                  double* st_icovs, int st_dim1, int st_dim2, int st_dim3,
                  double* st_mns, int st_mn_dim1, int st_mn_dim2,
                  double* st_icov_dets, int st_icov_dets_dim,
                  double* rangevec, int n)
{
  // ALLOCATE MEMORY
  int star_count = 0;
  int MAT_DIM = gr_dim1; //Typically set to 6
  int i, j, signum;
  double ApB_det, d_temp, result;
  gsl_permutation *p;

  gsl_matrix *A        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *B        = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_matrix *ApB      = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_vector *a        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *b        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *AapBb    = gsl_vector_alloc(MAT_DIM);
  gsl_vector *c        = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);
  gsl_vector *v_temp2  = gsl_vector_alloc(MAT_DIM);
  gsl_vector *amc      = gsl_vector_alloc(MAT_DIM); //will hold a - c
  gsl_vector *bmc      = gsl_vector_alloc(MAT_DIM); //will hold b - c

  p = gsl_permutation_alloc(A->size1);

  // INITIALISE GROUP MATRICES
  for (i=0; i<MAT_DIM; i++)
    for (j=0; j<MAT_DIM; j++)
      gsl_matrix_set (A, i, j, gr_icov[i*MAT_DIM + j]);

  for (i=0; i<MAT_DIM; i++)
    gsl_vector_set (a, i, gr_mn[i]);


  for (star_count=0; star_count<n; star_count++) {
    // INITIALISE STAR MATRICES
    for (i=0; i<MAT_DIM; i++)
      for (j=0; j<MAT_DIM; j++)
        gsl_matrix_set(B,i,j, st_icovs[star_count*MAT_DIM*MAT_DIM+i*MAT_DIM+j]);

    for (i=0; i<MAT_DIM; i++) 
      gsl_vector_set (b, i, st_mns[star_count*MAT_DIM + i]);


    // FIND OVERLAP
    // Adding A and B together and storing in ApB
    gsl_matrix_set_zero(ApB);
    gsl_matrix_add(ApB, A);
    gsl_matrix_add(ApB, B);
  
    // Storing the result A*a + B*b in AapBb
    gsl_vector_set_zero(AapBb);
    gsl_blas_dsymv(CblasUpper, 1.0, A, a, 1.0, AapBb);
    gsl_blas_dsymv(CblasUpper, 1.0, B, b, 1.0, AapBb);
  
    // Getting determinant of ApB
    gsl_linalg_LU_decomp(ApB, p, &signum);
    //ApB_det = gsl_linalg_LU_det(ApB, signum);
    ApB_det = fabs(gsl_linalg_LU_det(ApB, signum)); //temp doctoring determinant
  
    // Solve for c
    gsl_linalg_LU_solve(ApB, p, AapBb, c);
  
    // Compute the overlap formula
    gsl_vector_set_zero(v_temp);
    gsl_blas_dcopy(a, v_temp);       //v_temp holds a
    gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds a - c
    gsl_blas_dcopy(v_temp, amc);     //amc holds a - c
  
    // CAN'T HAVE v_temp and v_temp2 be the same vector.
    // Results in 0's being stored in v_temp2.
    gsl_blas_dsymv(CblasUpper, 1.0, A, v_temp, 0.0, v_temp2);
    //v_temp2 holds A (a-c)
  
    result = 0.0;
    gsl_blas_ddot(v_temp2, amc, &d_temp); //d_temp holds (a-c)^T A (a-c)
  
    result += d_temp;
    
    gsl_vector_set_zero(v_temp);
    gsl_blas_dcopy(b, v_temp);       //v_temp holds b
    gsl_blas_daxpy(-1.0, c, v_temp); //v_temp holds b - c
    gsl_blas_dcopy(v_temp, bmc);     //bmc holds b - c
  
    // CAN'T HAVE v_temp and v_temp2 be the same vector.
    // Results in 0's being stored in v_temp2.
    gsl_blas_dsymv(CblasUpper, 1.0, B, v_temp, 0.0, v_temp2);
    //v_temp2 holds A (b-c)
  
    gsl_blas_ddot(v_temp2, bmc, &d_temp); //d_temp holds (b-c)^T B (b-c)
    result += d_temp;
   
    result = -0.5 * result;
    result = exp(result);
  
    result *= sqrt((gr_icov_det * st_icov_dets[star_count]/ApB_det)
                  / pow(2*M_PI, MAT_DIM));

    // STORE IN 'rangevec'
    rangevec[star_count] = result;
  }

  // DEALLOCATE THE MEMORY
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_matrix_free(ApB);
  gsl_vector_free(a);
  gsl_vector_free(b);
  gsl_vector_free(AapBb);
  gsl_vector_free(c);
  gsl_vector_free(v_temp);
  gsl_vector_free(v_temp2);
  gsl_vector_free(amc);
  gsl_vector_free(bmc);

  gsl_permutation_free(p);
}

/* New main function, speed not yet tested
 * mostly here for debugging reasons
 * --parameters--
 *  group_icov     (6*6 npyArray) the group's inverse covariance matrix
 *  group_mn       (1*6 npyArray) which is the group's mean kinematic info
 *  group_icov_det (flt)          the determinent of the group_icov
 *  Bs             (nstars*6*6)   an array of each star's icov matrix
 *  bs:            (nstars*6)     an array of each star's mean kinematic info
 *  B_dets:        (nstars)       an array of the determinent of each icov
 *  nstars:        (int)          number of stars, used to determine the size
 *                          of npyArray which will return calculated overlaps)
 *
 * returns: (nstars) array of calculated overlaps of every star with 1 group
 *
 * todo: instead of calling internal function actually use cblas functions
 *          this will save time on the reallocation and deallocation
 *
 *      look up how to find inverse
 */
double new_get_lnoverlap(
  double* gr_cov, int gr_dim1, int gr_dim2,
  double* gr_mn, int gr_mn_dim,
  double* st_cov, int st_dim1, int st_dim2,
  double* st_mn, int st_mn_dim
  )
{
  //printf("Inside new_get_lnoverlaps function\n");
  printf("-----------------------------------------------------------\n");
  printf("new_get_lnoverlap(): In c implemntation of Tim's derivation\n");
  printf("-----------------------------------------------------------\n");
  //printf("Inputs are:\n");
  //printf("  A\n");
  //print_mat(gr_cov, gr_dim1, gr_dim2);
  //printf("  a\n");
  //print_vec(gr_mn, gr_mn_dim);
  //printf("  B\n");
  //print_mat(st_cov, st_dim1, st_dim2);
  //printf("  b\n");
  //print_vec(st_mn, st_mn_dim);

  // ALLOCATE MEMORY
  int MAT_DIM = gr_dim1; //Typically set to 6
  int i, j, signum;
  double d_temp, result, ln_det_BpA;
  FILE* fout = stdout;
  gsl_permutation *p1;

  gsl_matrix *BpA      = gsl_matrix_alloc(MAT_DIM, MAT_DIM); //(B+A)
  //gsl_matrix *BpAi     = gsl_matrix_alloc(MAT_DIM, MAT_DIM); //(B+A)^-1
  gsl_vector *bma      = gsl_vector_alloc(MAT_DIM); //will hold b - a
  gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);

  p1 = gsl_permutation_alloc(BpA->size1);

  printf("Memory allocated\n");
  // INITIALISE STAR MATRICES
  for (i=0; i<MAT_DIM; i++)
    for (j=0; j<MAT_DIM; j++)
      //perform B+A as part of the initialisation
      gsl_matrix_set(
        BpA,i,j,
        st_cov[i*MAT_DIM+j] +
        gr_cov[i*MAT_DIM+j]
      );
      printf("Printing BpA\n");
      print_matrix(fout, BpA);

  for (i=0; i<MAT_DIM; i++) {
    gsl_vector_set(
      bma, i,
      st_mn[i] -
      gr_mn[i]
    );
  }
  printf("Printing bma\n");
  print_vec(bma->data, 6);
  printf("Matrices initialised\n\n");

  result = 6*log(2*M_PI);
  printf("Added 6log(2pi):\n%6.2f\n", result);

  // Get inverse of BpA, this line is wrong, fix when have internet
  gsl_linalg_LU_decomp(BpA, p1, &signum);
  ln_det_BpA = log(fabs(gsl_linalg_LU_det(BpA, signum)));
  printf("Log of det(ApB): %6.2f\n", ln_det_BpA);
  result += ln_det_BpA;

  printf("result so far:\n%6.2f\n",result);
  //
  gsl_vector_set_zero(v_temp);
  gsl_linalg_LU_solve(BpA, p1, bma, v_temp); //v_temp holds (B+A)^-1 (b-a)
  gsl_blas_ddot(v_temp, bma, &d_temp); //d_temp holds (b-a)^T (B+A)-1 (b-a)
  printf("Printing bma_BpAi_bma\n");
  printf("%6.2f\n\n", d_temp);

  result += d_temp;
  printf("result after bma_BpAi_bma:\n%6.2f\n",result);

  result *= -0.5;
  printf("Everything calculated\n");
  printf("Final result:\n%6.8f\n", result);
  //

  // DEALLOCATE THE MEMORY
  gsl_matrix_free(BpA);
  //gsl_matrix_free(BpAi);
  gsl_vector_free(bma);
  gsl_vector_free(v_temp);

  gsl_permutation_free(p1);

  return result;
  //printf("At end of new_get_lnoverlaps function\n");
}



