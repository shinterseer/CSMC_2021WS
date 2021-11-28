
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


  for (size_t row=0; row < N; ++row) {
    double val = 0; // y = Ax for this row
    for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row+1]; ++jj) {
      val += csr_values[jj] * x[csr_colindices[jj]];
    }
    y[row] = val;
  }

}


/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
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

    // similarly for the other operations

    // lines 5,6:
    double res_norm = 0;
    for (size_t i=0; i<N; ++i) res_norm += r[i] * r[i];
    double alpha = 0;
    for (size_t i=0; i<N; ++i) alpha += p[i] * Ap[i];
    alpha = res_norm / alpha;
    std::cout << "alpha: " << alpha << std::endl;

    // line 7,8:
    for (size_t i=0; i<N; ++i) {
      solution[i] += alpha *  p[i];
      r[i]        -= alpha * Ap[i];
    }

    double beta = res_norm;

    // lines 9, 10:
    res_norm = 0;
    for (size_t i=0; i<N; ++i) res_norm += r[i] * r[i];
    std::cout << "residual_norm_squared: " << res_norm << std::endl;
    if (res_norm < 1e-7) break;

    // line 11: compute beta
    beta = res_norm / beta;
    std::cout << "beta: " << beta << std::endl;

    // line 12: update p
    for (size_t i=0; i<N; ++i) p[i] = r[i] + beta * p[i];

    if (iters > 100) break;  // solver didn't converge
    ++iters;

    std::cout << "------" << std::endl;
  }

  std::cout << "Conjugate Gradients converged in " << iters << " iterations." << std::endl;
}



/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction) {

  size_t N = points_per_direction * points_per_direction; // number of unknows to solve for

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

  //
  // Call Conjugate Gradient implementation
  //
  conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6)" << std::endl;
}


int main() {

  solve_system(100); // solves a system with 100*100 unknowns

  return EXIT_SUCCESS;
}