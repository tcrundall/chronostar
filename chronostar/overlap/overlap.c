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
void inplace(double* npyArray3D, int npyLength1D, int npylength2D,
              int npylength2D, double* invec, int n)
{
  int i;
  for (i=0; i<n; i++) {
    invec[i] = npyArray3D[i];
  }
}
*/

int print_matrix(FILE *f, const gsl_matrix *m)
{
  int status, n = 0;

  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
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

int print_vector(FILE *f, const gsl_vector *m)
{
  int status, n = 0;

  for (size_t i = 0; i < m->size; i++) {
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

int sum(int* npyArray3D, int npyLength1D, int npyLength2D, int npyLength3D)
{
  int i, j, k;
  int sum = 0;

  for (i=0; i<npyLength1D; i++)
    for (j=0; j<npyLength2D; j++)
      for (k=0; k<npyLength3D; k++)
        sum += npyArray3D[i*npyLength3D*npyLength2D + k*npyLength2D + j];

  return sum;
}

double get_det(PyObject *A)
{
  int MAT_DIM = 6;
  int i, signum;
  double det;
  int nInts = PyList_Size(A);

  gsl_matrix *m = gsl_matrix_alloc(MAT_DIM, MAT_DIM);
  gsl_permutation *p;

  p = gsl_permutation_alloc(m->size1);
  
  for (i=0; i<nInts; i++)
  {
    PyObject *oo = PyList_GetItem(A, i);
    //printf("%6.2f\n",PyFloat_AS_DOUBLE(oo));
    gsl_matrix_set (m, i%MAT_DIM, i/MAT_DIM, PyFloat_AS_DOUBLE(oo));
  }
  
  gsl_linalg_LU_decomp(m, p, &signum);
  det = gsl_linalg_LU_det(m, signum);
  //printf("%6.2f\n",det);
  return det;
}

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
 *      look up how to access math.pi
 */
void new_get_lnoverlaps(
  double* gr_cov, int gr_dim1, int gr_dim2,
  double* gr_mn, int gr_mn_dim,
  double* st_covs, int st_dim1, int st_dim2, int st_dim3,
  double* st_mns, int st_mn_dim1, int st_mn_dim2,
  double* rangevec, int n
  )
{
  //printf("Inside new_get_lnoverlaps function\n");
  // ALLOCATE MEMORY
  int star_count = 0;
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

  //printf("Memory allocated\n");
  for (star_count=0; star_count<n; star_count++) {
    // INITIALISE STAR MATRICES
    for (i=0; i<MAT_DIM; i++)
      for (j=0; j<MAT_DIM; j++)
        //perform B+A as part of the initialisation
        gsl_matrix_set(
          BpA,i,j,
          st_covs[star_count*MAT_DIM*MAT_DIM+i*MAT_DIM+j] + 
          gr_cov[i*MAT_DIM+j]
        );
//    printf("Printing BpA\n");
//    print_matrix(fout, BpA);

    for (i=0; i<MAT_DIM; i++) {
      gsl_vector_set(
        bma, i,
        st_mns[star_count*MAT_DIM + i] - 
        gr_mn[i]
      );
    }
    //printf("Printing bma\n");
    //print_vec(bma->data, 6);
    //printf("Matrices initialised\n\n");

    result = 6*log(2*M_PI);
    // To Do! put 6ln(2 pi) in here ^^

    // Get inverse of BpA, this line is wrong, fix when have internet
    gsl_linalg_LU_decomp(BpA, p1, &signum);
    ln_det_BpA = log(fabs(gsl_linalg_LU_det(BpA, signum)));
    result += ln_det_BpA;

    //printf("ln(det(BpA)) added\n");
    //printf("%6.2f\n\n",ln_det_BpA);
    //printf("result so far:\n%6.2f\n",result);
    //// Solve for c
    //gsl_linalg_LU_solve(ApB, p, AapBb, c);
    
    // Don't use invert, use solve like example above ^^
    //gsl_linalg_LU_invert(BpA, p1, BpAi);
    //
    gsl_vector_set_zero(v_temp);
    gsl_linalg_LU_solve(BpA, p1, bma, v_temp); //v_temp holds (B+A)^-1 (b-a)
    gsl_blas_ddot(v_temp, bma, &d_temp); //d_temp holds (b-a)^T (B+A)-1 (b-a)
    //printf("Printing bma_BpAi_bma\n");
    //printf("%6.2f\n\n", d_temp);

    result += d_temp;
    //printf("result after bma_BpAi_bma:\n%6.2f\n",result);

    result *= -0.5;
    //printf("Everything calculated\n");
    //printf("Final result:\n%6.2f\n", result);
    //
    // STORE IN 'rangevec'
    rangevec[star_count] = result;
  }

  // DEALLOCATE THE MEMORY
  gsl_matrix_free(BpA);
  //gsl_matrix_free(BpAi);
  gsl_vector_free(bma);
  gsl_vector_free(v_temp);

  gsl_permutation_free(p1);

  //printf("At end of new_get_lnoverlaps function\n");
}
