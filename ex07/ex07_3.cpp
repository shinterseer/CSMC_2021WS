//
// Tutorial for demonstrating a simple OpenCL vector addition kernel
//
// Author: Karl Rupp    rupp@iue.tuwien.ac.at
//

typedef double       ScalarType;


#include <iostream>
#include <iomanip> // std::setprecision()
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Helper include file for error checking
#include "ocl-error.hpp"
#include "timer.hpp"

#include <numeric> // for std::accumulate
#include <algorithm> // for std::sort
#include "generate.hpp" // generate sparse matrices



#define BLOCK_SIZE 1
#define GRID_SIZE 1


// const char *my_opencl_program = ""
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"    // required to enable 'double' inside OpenCL programs
// ""
// "__kernel void vec_add(__global double *x,\n"
// "                      __global double *y,\n"
// "                      __global double *result,\n"
// "                      unsigned int N\n)"
// "{\n"
// "	__local double shared_dotp[256];\n"
// "  double thread_dotp = 0;\n"
// "  for (unsigned int i  = get_global_id(0);\n"
// "                    i  < N;\n"
// "                    i += get_global_size(0))\n"
// "    thread_dotp += x[i] * y[i];\n"
// "	shared_dotp[get_local_id(0)] = thread_dotp;\n"
// "	// now the reduction\n"
// "	for(int stride = get_local_size(0)/2; stride>0; stride/=2){\n"
// "		barrier(CLK_GLOBAL_MEM_FENCE);\n"
// "		if (get_local_id(0) < stride){\n"
// "			shared_dotp[get_local_id(0)] += shared_dotp[get_local_id(0) + stride];\n"
// "		}\n"
// "	}\n"
// "	barrier(CLK_GLOBAL_MEM_FENCE);	\n"	
// "    if (get_local_id(0) == 0)\n"
// "		   result[get_group_id(0)] = shared_dotp[0];\n"
// "}";  // you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.


const char *my_opencl_program = ""
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"    // required to enable 'double' inside OpenCL programs
""
"__kernel void vec_add( unsigned int N,\n"
"                       __global int *csr_rowoffsets,\n"
"                       __global int *csr_colindices,\n"
"                       __global double *csr_values,\n"
"                       __global double *vector,\n"
"                       __global double *result)\n"
"{\n"
"  for (size_t i=0; i<N; ++i) {\n"
"    double value = 0;\n"
"    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; ++j)\n"
"      value += csr_values[j] * vector[csr_colindices[j]];\n"
"\n"
"    result[i] = value;\n"
"  }\n"
"}";  // you can have multiple kernels within a single OpenCL program. For simplicity, this OpenCL program contains only a single kernel.




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


// returns max(abs(v1 - v2))
double compare_vectors(std::vector<double> v1, std::vector<double> v2)
{
	double max = 0;
  for (size_t i=0; i < v1.size(); ++i)
		if (std::fabs(v1[i] - v2[i]) > max)
			max = std::fabs(v1[i] - v2[i]);
	return max;
}



// ___________________________ the big Function ___________________________
// ________________________________________________________________________


int benchmark_matvec(size_t points_per_direction, size_t max_nonzeros_per_row,
                      void (*generate_matrix)(size_t, int*, int*, double*)) // function pointer parameter
{
	
	size_t grid_size = GRID_SIZE;
  size_t local_size = BLOCK_SIZE;
	size_t global_size = GRID_SIZE * BLOCK_SIZE;

	bool compute_on_gpu = false;
	bool sanity_check = true;
	int repetitions = 1;
	bool dotp = false;


  //
  /////////////////////////// Part 1: Set up an OpenCL context with one device ///////////////////////////////////
  //
  cl_int err;

  //
  // Query platform:
  //
  cl_uint num_platforms;
  cl_platform_id platform_ids[42];   //no more than 42 platforms supported...
  err = clGetPlatformIDs(42, platform_ids, &num_platforms); OPENCL_ERR_CHECK(err);
  // std::cout << "# Platforms found: " << num_platforms << std::endl;
  cl_platform_id my_platform = platform_ids[0];
	if (compute_on_gpu)
		my_platform = platform_ids[1];

  //
  // Query devices:
  //
  cl_device_id device_ids[42];
  cl_uint num_devices;
  err = clGetDeviceIDs(my_platform, CL_DEVICE_TYPE_ALL, 42, device_ids, &num_devices); OPENCL_ERR_CHECK(err);
  // std::cout << "# Devices found: " << num_devices << std::endl;
  cl_device_id my_device_id = device_ids[0];

  char device_name[64];
  size_t device_name_len = 0;
  err = clGetDeviceInfo(my_device_id, CL_DEVICE_NAME, sizeof(char)*63, device_name, &device_name_len); OPENCL_ERR_CHECK(err);
  // std::cout << "Using the following device: " << device_name << std::endl;

  //
  // Create context:
  //
  cl_context my_context = clCreateContext(0, 1, &my_device_id, NULL, NULL, &err); OPENCL_ERR_CHECK(err);


  //
  // create a command queue for the device:
  //
  cl_command_queue my_queue = clCreateCommandQueueWithProperties(my_context, my_device_id, 0, &err); OPENCL_ERR_CHECK(err);



  //
  /////////////////////////// Part 2: Create a program and extract kernels ///////////////////////////////////
  //

  Timer timer;
  // timer.reset();

  //
  // Build the program:
  //
  size_t source_len = std::string(my_opencl_program).length();
  cl_program prog = clCreateProgramWithSource(my_context, 1, &my_opencl_program, &source_len, &err);OPENCL_ERR_CHECK(err);
  err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

  //
  // Print compiler errors if there was a problem:
  //
  if (err != CL_SUCCESS) {

    char *build_log;
    size_t ret_val_size;
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    build_log = (char *)malloc(sizeof(char) * (ret_val_size+1));
    err = clGetProgramBuildInfo(prog, my_device_id, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    build_log[ret_val_size] = '\0'; // terminate string
    std::cout << "Log: " << build_log << std::endl;
    free(build_log);
    std::cout << "OpenCL program sources: " << std::endl << my_opencl_program << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Extract the only kernel in the program:
  //
  cl_kernel my_kernel = clCreateKernel(prog, "vec_add", &err); OPENCL_ERR_CHECK(err);
  // std::cout << "Time to compile and create kernel: " << timer.get() << std::endl;

		
		
	//
	/////////////////////////// Part 3: Create memory buffers ///////////////////////////////////
	//

	//
	// Set up buffers on host:
	//

	cl_uint vector_size = points_per_direction * points_per_direction;
	std::vector<ScalarType> x(vector_size, 2.0);
	std::vector<ScalarType> y(vector_size, 3.0);
	std::vector<ScalarType> result(grid_size, 0.0);


	std::vector<int> csr_rowoffsets(vector_size+1,0);
	std::vector<int> csr_colindices(max_nonzeros_per_row*vector_size,0);
	std::vector<double> csr_values(max_nonzeros_per_row*vector_size,0);
	std::vector<double> x_vector(vector_size,1);
	std::vector<double> result_vector(vector_size,0);
	std::vector<double> result_vector_sanity(vector_size,0);
	

	//
	// fill CSR matrix with values
	//
	generate_fdm_laplace(points_per_direction, &csr_rowoffsets[0], &csr_colindices[0], &csr_values[0]);		
		
	//
	// Now set up OpenCL buffers:
	//
	cl_mem ocl_csr_rowoffsets = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (vector_size+1) * sizeof(int), &(csr_rowoffsets[0]), &err); OPENCL_ERR_CHECK(err);
	cl_mem ocl_csr_colindices = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_nonzeros_per_row * vector_size * sizeof(int), &(csr_colindices[0]), &err); OPENCL_ERR_CHECK(err);
	cl_mem ocl_csr_values = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, max_nonzeros_per_row * vector_size * sizeof(double), &(csr_values[0]), &err); OPENCL_ERR_CHECK(err);
	cl_mem ocl_x_vector = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(double), &(x_vector[0]), &err); OPENCL_ERR_CHECK(err);
	cl_mem ocl_result_vector = clCreateBuffer(my_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vector_size * sizeof(double), &(result_vector[0]), &err); OPENCL_ERR_CHECK(err);


	//
	/////////////////////////// Part 4: Run kernel ///////////////////////////////////
	//

	//
	// Set kernel arguments:
	//
	err = clSetKernelArg(my_kernel, 0, sizeof(cl_uint), (void*)&vector_size); OPENCL_ERR_CHECK(err);		
	err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem),  (void*)&ocl_csr_rowoffsets); OPENCL_ERR_CHECK(err);
	err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem),  (void*)&ocl_csr_colindices); OPENCL_ERR_CHECK(err);
	err = clSetKernelArg(my_kernel, 3, sizeof(cl_mem),  (void*)&ocl_csr_values); OPENCL_ERR_CHECK(err);
	err = clSetKernelArg(my_kernel, 4, sizeof(cl_mem),  (void*)&ocl_x_vector); OPENCL_ERR_CHECK(err);
	err = clSetKernelArg(my_kernel, 5, sizeof(cl_mem),  (void*)&ocl_result_vector); OPENCL_ERR_CHECK(err);		


	//
	// Enqueue kernel in command queue:
	//
	std::vector<double> execution_times;
	for(int i = 0; i < repetitions; i++){
		timer.reset();
		
		err = clEnqueueNDRangeKernel(my_queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); OPENCL_ERR_CHECK(err);

		// wait for all operations in queue to finish:
		err = clFinish(my_queue); OPENCL_ERR_CHECK(err);
		execution_times.push_back(timer.get());	
	}
	std::sort(execution_times.begin(), execution_times.end());
	double median_time = execution_times[int(repetitions/2)];
	std::cout << std::setprecision(8) << std::scientific << vector_size << "; " << median_time << std::endl;

	//
	/////////////////////////// Part 5: Get data from OpenCL buffer ///////////////////////////////////
	//

	err = clEnqueueReadBuffer(my_queue, ocl_result_vector, CL_TRUE, 0, sizeof(double) * result_vector.size(), &(result_vector[0]), 0, NULL, NULL); OPENCL_ERR_CHECK(err);

	

	if (sanity_check){
		// calculate result on cpu and compare
		csr_matvec_product(vector_size, &csr_rowoffsets[0], &csr_colindices[0], &csr_values[0], 
											 &x_vector[0], &result_vector_sanity[0]);
		std::cout << std::endl;
		std::cout << "vector comparison (max(abs(result_gpu - result_cpu))):" << std::endl;
		std::cout << compare_vectors(result_vector, result_vector_sanity) << std::endl;
	}

	//
	// cleanup
	//

	// no need to release host vectors because std::vector was used
	clReleaseMemObject(ocl_csr_rowoffsets);
	clReleaseMemObject(ocl_csr_colindices);
	clReleaseMemObject(ocl_csr_values);
	clReleaseMemObject(ocl_x_vector);
	clReleaseMemObject(ocl_result_vector);

	
  clReleaseProgram(prog);
  clReleaseCommandQueue(my_queue);
  clReleaseContext(my_context);
  return EXIT_SUCCESS;
	
}

// ___________________________ Main _______________________________________
// ________________________________________________________________________

int main()
{

	
	std::vector<size_t> points_per_direction = {30, 60, 120};	
	std::cout << "vector size; execution time" << std::endl;

	for(size_t i = 0; i < points_per_direction.size(); ++i){
		benchmark_matvec(points_per_direction[i],5,generate_fdm_laplace);
	}
	
	
  return EXIT_SUCCESS;
}

