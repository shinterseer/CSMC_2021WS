
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"


/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
void csr_matvec_product(size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *x, double *y)
{
  /** YOUR CODE HERE
   *
   *  Either provide a CPU implementation, or call the CUDA kernel here
   *
   */

}


/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use with CUDA.
 *  Modify as you see fit.
 */
void conjugate_gradient(size_t N,  // number of unknows
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *rhs,
                        double *solution)
                        //, double *init_guess)   // feel free to add a nonzero initial guess as needed
{

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double *p = (double*)malloc(sizeof(double) * N);
  double *r = (double*)malloc(sizeof(double) * N);
  double *Ap = (double*)malloc(sizeof(double) * N);

  // line 2: initialize r and p:
  std::copy(rhs, rhs+N, p);
  std::copy(rhs, rhs+N, r);

  int iters = 0;
  while (1) {

    // line 4: A*p:
    csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, p, Ap);


    /** YOUR CODE HERE
    *
    * similarly for the other operations
    *
    */

    if (iters > 10000) break;  // solver didn't converge
    ++iters;
  }

  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations" << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations." << std::endl;

  free(p);
  free(r);
  free(Ap);
}



/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction) {

  size_t N = points_per_direction * points_per_direction; // number of unknows to solve for

  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  int *csr_colindices =    (int*)malloc(sizeof(double) * 5 * N);
  double *csr_values  = (double*)malloc(sizeof(double) * 5 * N);

  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double*)malloc(sizeof(double) * N);
  double *rhs      = (double*)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);

  /**
   *
   * YOUR CODE HERE: Allocate GPU arrays as needed
   *
   **/

  //
  // Call Conjugate Gradient implementation (CPU arrays passed here; modify to use GPU arrays)
  //
  conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6)" << std::endl;

  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}


int main() {

  solve_system(100); // solves a system with 100*100 unknowns

  return EXIT_SUCCESS;
}
