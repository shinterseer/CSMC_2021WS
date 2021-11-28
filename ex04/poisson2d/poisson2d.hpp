
#include <vector>
#include <map>
#include <cmath>


/** @brief Generates the system matrix for a 2D finite difference discretization of the heat equation
 *    -\Delta u = 1
 * on a square domain with homogeneous boundary conditions.
 *
 * Parameters:
 *   - points_per_direction: The number of discretization points in x- and y-direction (square domain)
 *   - csr_rowoffsets, csr_colindices, csr_values: CSR arrays. 'rowoffsets' is the offset aray, 'colindices' holds the 0-based column-indices, 'values' holds the nonzero values.
 */
void generate_fdm_laplace(size_t points_per_direction,
                          int *csr_rowoffsets, int *csr_colindices, double *csr_values)
{
  size_t total_unknowns = points_per_direction * points_per_direction;

  //
  // set up the system matrix using easy-to-use STL types:
  //
  std::vector<std::map<int, double> > A(total_unknowns);

  for (size_t i=0; i<points_per_direction; ++i)
  {
    for (size_t j=0; j<points_per_direction; ++j)
    {
      size_t row = i + j * points_per_direction;

      A[row][row] = 4.0;

      if (i > 0)
      {
        size_t col = (i-1) + j * points_per_direction;
        A[row][col] = -1.0;
      }

      if (j > 0)
      {
        size_t col = i + (j-1) * points_per_direction;
        A[row][col] = -1.0;
      }

      if (i < points_per_direction-1)
      {
        size_t col = (i+1) + j * points_per_direction;
        A[row][col] = -1.0;
      }

      if (j < points_per_direction-1)
      {
        size_t col = i + (j+1) * points_per_direction;
        A[row][col] = -1.0;
      }
    }
  }

  //
  // write data to CSR arrays
  //
  size_t k = 0; // index within column and value arrays
  for (size_t row=0; row < A.size(); ++row) {
    csr_rowoffsets[row] = k;
    for (std::map<int, double>::const_iterator it = A[row].begin(); it != A[row].end(); ++it) {
      csr_colindices[k] = it->first;
      csr_values[k] = it->second;
      ++k;
    }
  }
  csr_rowoffsets[A.size()] = k;

}

/** @brief Helper routine to compute ||Ax - b||_2. */
double relative_residual(size_t N,
                       int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                       double *rhs, double *solution)
{
  // compute ||b|| as the initial residual with guess 'x = 0'.
  double rhs_norm = 0;
  for (size_t i=0; i<N; ++i) rhs_norm += rhs[i] * rhs[i];

  // compute ||Ax - b||_2
  double residual_norm = 0;
  for (size_t row=0; row < N; ++row) {
    double y = 0; // y = Ax for this row
    for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row+1]; ++jj) {
      y += csr_values[jj] * solution[csr_colindices[jj]];
    }
    residual_norm += (y-rhs[row]) * (y-rhs[row]);
  }

  return std::sqrt(residual_norm / rhs_norm);
}
