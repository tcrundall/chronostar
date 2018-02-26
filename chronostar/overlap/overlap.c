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

void get_lnoverlaps(
  double* gr_cov, int gr_dim1, int gr_dim2,
  double* gr_mn, int gr_mn_dim,
  double* st_covs, int st_dim1, int st_dim2, int st_dim3,
  double* st_mns, int st_mn_dim1, int st_mn_dim2,
  double* lnoverlaps_vec, int n
  )
  /* Calculates the natural log overlaps with the 6D Gaussians described by the star mean and cov matrices
   * with the 6D Gaussian decribed by the group mean and cov matrix
   */
{
  // ALLOCATE MEMORY
  int star_count;
  int MAT_DIM = gr_dim1; //Typically set to 6
  int i, j, signum;
  double d_temp, ln_det_BpA;
  double result;

  gsl_permutation *p;
  gsl_matrix *BpA      = gsl_matrix_alloc(MAT_DIM, MAT_DIM); //(B+A)
  gsl_vector *bma      = gsl_vector_alloc(MAT_DIM); //will hold b - a
  gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);

  p = gsl_permutation_alloc(BpA->size1);

  //printf("Memory allocated\n");
  for (star_count=0; star_count<n; star_count++) {
    result = 0.0;
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

    result += 6*log(2*M_PI);

    // Get inverse of BpA, this line is wrong, fix when have internet
    gsl_linalg_LU_decomp(BpA, p, &signum);
    ln_det_BpA = log(fabs(gsl_linalg_LU_det(BpA, signum)));
    result += ln_det_BpA;
    
    // Don't find inverse of BpA - solve for c in (B+A)c = (b-a)
    gsl_vector_set_zero(v_temp);
    gsl_linalg_LU_solve(BpA, p, bma, v_temp); //v_temp holds (B+A)^-1 (b-a)
    gsl_blas_ddot(v_temp, bma, &d_temp); //d_temp holds (b-a)^T (B+A)-1 (b-a)

    result += d_temp;
    //printf("result after bma_BpAi_bma:\n%6.2f\n",result);

    result *= -0.5;

    // STORE IN 'lnoverlaps_vec', the return vector
    lnoverlaps_vec[star_count] = result;
  }

  // DEALLOCATE THE MEMORY
  gsl_matrix_free(BpA);
  gsl_vector_free(bma);
  gsl_vector_free(v_temp);

  gsl_permutation_free(p);
}
