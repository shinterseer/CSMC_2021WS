
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>

/** @brief Helper routine to convert a STL-typed matrix to CSR arrays */
void convert_to_CSR(std::vector<std::map<int, double> > & A,
                    int *csr_rowoffsets, int *csr_colindices, double *csr_values)
{
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


/** @brief Generates a ex7-specific matrix with about 200 to 2000 nonzero entries per row
 *
 * Parameters:
 *   - points_per_direction: The number of discretization points in x- and y-direction (square domain)
 *   - csr_rowoffsets, csr_colindices, csr_values: CSR arrays. 'rowoffsets' is the offset aray, 'colindices' holds the 0-based column-indices, 'values' holds the nonzero values.
 */
void generate_matrix2(size_t points_per_direction,
                      int *csr_rowoffsets, int *csr_colindices, double *csr_values)
{
  size_t total_unknowns = points_per_direction * points_per_direction;

  //
  // set up the system matrix using easy-to-use STL types:
  //
  std::vector<std::map<int, double> > A(total_unknowns);

  srand(0); // make
  for (size_t row=0; row<total_unknowns; ++row)
  {
    std::map<int, double> & A_row = A[row];
    A_row[row] = 4.0;

    size_t num_entries = 200 + rand() % 1800;
    for (size_t j=0; j<num_entries; ++j) {
      size_t col = rand() % total_unknowns;
      A_row[col] = -1;
    }
  }

  convert_to_CSR(A, csr_rowoffsets, csr_colindices, csr_values);
}



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

  convert_to_CSR(A, csr_rowoffsets, csr_colindices, csr_values);
}

