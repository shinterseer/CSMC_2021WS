STRINGIFY( CUCL_KERNEL void dotProduct(
  CUCL_GLOBALMEM double *x,
  CUCL_GLOBALMEM double *y,
  CUCL_GLOBALMEM double * partial_result,
  unsigned int N)
{
  CUCL_LOCALMEM double shared_buf[512]; double thread_sum = 0;
  for (int i = CUCL_GLOBALID0; i < N; i += CUCL_GLOBALSIZE0)
    thread_sum += x[i]* y[i];

  shared_buf[CUCL_LOCALID0] = thread_sum;
  for (int stride = CUCL_LOCALSIZE0 / 2; stride > 0; stride /= 2) {
    CUCL_BARRIER;
    if (CUCL_LOCALID0 < stride)
      shared_buf[CUCL_LOCALID0] += shared_buf[CUCL_LOCALID0+stride];
  }

  CUCL_BARRIER;
  if (CUCL_LOCALID0 == 0)
    partial_result[CUCL_GROUPID0] = shared_buf[0];
} )


