
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "generate.hpp"
#include "timer.hpp"


/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y. CPU implementation.  */
void csr_matvec_product(size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double const *x, double *y)
{
  for (size_t i=0; i<N; ++i) {
    double value = 0;
    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; ++j)
      value += csr_values[j] * x[csr_colindices[j]];

    y[i] = value;
  }

}


/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void benchmark_matvec(size_t points_per_direction, size_t max_nonzeros_per_row,
                      void (*generate_matrix)(size_t, int*, int*, double*)) // function pointer parameter
{

  size_t N = points_per_direction * points_per_direction; // number of rows and columns

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
  int *csr_colindices =    (int*)malloc(sizeof(double) * max_nonzeros_per_row * N);
  double *csr_values  = (double*)malloc(sizeof(double) * max_nonzeros_per_row * N);

  //
  // fill CSR matrix with values
  //
  generate_matrix(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

  //
  // Allocate vectors:
  //
  double *x = (double*)malloc(sizeof(double) * N); std::fill(x, x + N, 1);
  double *y = (double*)malloc(sizeof(double) * N); std::fill(y, y + N, 0);


  //
  // Call matrix-vector product kernel
  //
  Timer timer;
  timer.reset();
  csr_matvec_product(N, csr_rowoffsets, csr_colindices, csr_values, x, y);
  std::cout << "Time for product: " << timer.get() << std::endl;

  free(x);
  free(y);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}


int main() {

  std::cout << "# Benchmarking finite difference matrix" << std::endl;
  benchmark_matvec(100, 5, generate_fdm_laplace); // 100*100 unknowns, finite difference matrix

  std::cout << "# Benchmarking special matrix" << std::endl;
  benchmark_matvec(100, 2000, generate_matrix2);     // 100*100 unknowns, special matrix with 200-2000 nonzeros per row

  return EXIT_SUCCESS;
}
