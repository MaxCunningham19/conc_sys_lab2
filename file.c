/* Test and timing harness program for developing a multichannel
   multikernel convolution (as used in deep learning networks)

   Note there are some simplifications around this implementation,
   in particular with respect to computing the convolution at edge
   pixels of the image.

   Author: David Gregg
   Date:   March 2022

   Version 1.7 : Adjusted types for mixed-type computation

   Version 1.6 : Modified the code so that the input tensor is float

   Version 1.5 : Modified the code so that the input and kernel
                 are tensors of 16-bit integer values

   Version 1.4 : Modified the random generator to reduce the range
                 of generated values;

   Version 1.3 : Fixed which loop variables were being incremented
                 in write_out();
                 Fixed dimensions of output and control_output
                 matrices in main function

   Version 1.2 : Changed distribution of test data to (hopefully)
                 eliminate random walk of floating point error;
                 Also introduced checks to restrict kernel-order to
                 a small set of values

   Version 1.1 : Fixed bug in code to create 4d matrix
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <stdint.h>
#include <x86intrin.h>

const int MIN = 256;

/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

/* write 3d matrix to stdout */
void write_out(int16_t ***a, int dim0, int dim1, int dim2)
{
  int i, j, k;

  for (i = 0; i < dim0; i++)
  {
    printf("Outer dimension number %d\n", i);
    for (j = 0; j < dim1; j++)
    {
      for (k = 0; k < dim2 - 1; k++)
      {
        printf("%d, ", a[i][j][k]);
      }
      // print end of line
      printf("%f\n", a[i][j][dim2 - 1]);
    }
  }
}

/* create new empty 4d float matrix */
float ****new_empty_4d_matrix_float(int dim0, int dim1, int dim2, int dim3)
{
  float ****result = malloc(dim0 * sizeof(float ***));
  float ***mat1 = malloc(dim0 * dim1 * sizeof(float **));
  float **mat2 = malloc(dim0 * dim1 * dim2 * sizeof(float *));
  float *mat3 = malloc(dim0 * dim1 * dim2 * dim3 * sizeof(float));
  int i, j, k;

  for (i = 0; i < dim0; i++)
  {
    result[i] = &(mat1[i * dim1]);
    for (j = 0; j < dim1; j++)
    {
      result[i][j] = &(mat2[i * dim1 * dim2 + j * dim2]);
      for (k = 0; k < dim2; k++)
      {
        result[i][j][k] = &(mat3[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3]);
      }
    }
  }

  return result;
}

/* create new empty 3d matrix */
float ***new_empty_3d_matrix_float(int dim0, int dim1, int dim2)
{
  float ****mat4d;
  float ***mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix_float(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* create new empty 4d int16_t matrix */
int16_t ****new_empty_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
  int16_t ****result = malloc(dim0 * sizeof(int16_t ***));
  int16_t ***mat1 = malloc(dim0 * dim1 * sizeof(int16_t **));
  int16_t **mat2 = malloc(dim0 * dim1 * dim2 * sizeof(int16_t *));
  int16_t *mat3 = malloc(dim0 * dim1 * dim2 * dim3 * sizeof(int16_t));
  int i, j, k;

  for (i = 0; i < dim0; i++)
  {
    result[i] = &(mat1[i * dim1]);
    for (j = 0; j < dim1; j++)
    {
      result[i][j] = &(mat2[i * dim1 * dim2 + j * dim2]);
      for (k = 0; k < dim2; k++)
      {
        result[i][j][k] = &(mat3[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3]);
      }
    }
  }

  return result;
}

/* create new empty 3d matrix */
int16_t ***new_empty_3d_matrix_int16(int dim0, int dim1, int dim2)
{
  int16_t ****mat4d;
  int16_t ***mat3d;

  // create a 4d matrix with single first dimension
  mat4d = new_empty_4d_matrix_int16(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* take a copy of the matrix and return in a newly allocated matrix */
int16_t ****copy_4d_matrix(int16_t ****source_matrix, int dim0,
                           int dim1, int dim2, int dim3)
{
  int i, j, k, l;
  int16_t ****result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

  for (i = 0; i < dim0; i++)
  {
    for (j = 0; j < dim1; j++)
    {
      for (k = 0; k < dim2; k++)
      {
        for (l = 0; l < dim3; l++)
        {
          result[i][j][k][l] = source_matrix[i][j][k][l];
        }
      }
    }
  }
  return result;
}

/* create a matrix and fill it with random numbers */
int16_t ****gen_random_4d_matrix_int16(int dim0, int dim1, int dim2, int dim3)
{
  int16_t ****result;
  int i, j, k, l;
  struct timeval seedtime;
  int seed;

  result = new_empty_4d_matrix_int16(dim0, dim1, dim2, dim3);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  const int range = 1 << 10; // 2^10
  // const int bias = 1 << 16; // 2^16
  int16_t offset = 0.0;
  for (i = 0; i < dim0; i++)
  {
    for (j = 0; j < dim1; j++)
    {
      for (k = 0; k < dim2; k++)
      {
        for (l = 0; l < dim3; l++)
        {
          // generate uniform random integer with mean of zero
          long long rand = random();
          // now cut down the range and bias the mean to reduce
          // the likelihood of large floating point round-off errors
          int reduced_range = (rand % range);
          result[i][j][k][l] = reduced_range;
        }
      }
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
float ****gen_random_4d_matrix_float(int dim0, int dim1, int dim2, int dim3)
{
  float ****result;
  int i, j, k, l;
  struct timeval seedtime;
  int seed;

  result = new_empty_4d_matrix_float(dim0, dim1, dim2, dim3);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  const int range = 1 << 12; // 2^12
  const int bias = 1 << 10;  // 2^16
  int16_t offset = 0.0;
  for (i = 0; i < dim0; i++)
  {
    for (j = 0; j < dim1; j++)
    {
      for (k = 0; k < dim2; k++)
      {
        for (l = 0; l < dim3; l++)
        {
          // generate uniform random integer with mean of zero
          long long rand = random();
          // now cut down the range and bias the mean to reduce
          // the likelihood of large floating point round-off errors
          int reduced_range = (rand % range);
          result[i][j][k][l] = reduced_range + bias;
        }
      }
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
float ***gen_random_3d_matrix_float(int dim0, int dim1, int dim2)
{
  float ****mat4d;
  float ***mat3d;

  // create a 4d matrix with single first dimension
  mat4d = gen_random_4d_matrix_float(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* create a matrix and fill it with random numbers */
int16_t ***gen_random_3d_matrix_int16(int dim0, int dim1, int dim2)
{
  int16_t ****mat4d;
  int16_t ***mat3d;

  // create a 4d matrix with single first dimension
  mat4d = gen_random_4d_matrix_int16(1, dim0, dim1, dim2);
  // now throw away out first dimension
  mat3d = mat4d[0];
  free(mat4d);
  return mat3d;
}

/* check the sum of absolute differences is within reasonable epsilon */
void check_result(float ***result, float ***control,
                  int dim0, int dim1, int dim2)
{
  int i, j, k;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  // printf("SAD\n");

  for (i = 0; i < dim0; i++)
  {
    for (j = 0; j < dim1; j++)
    {
      for (k = 0; k < dim2; k++)
      {
        float control_check = control[i][j][k];
        float result_check = result[i][j][k];
        //printf("%f", control_check);
        //printf("%f", result_check);
        double diff = fabs(control[i][j][k] - result[i][j][k]);
        assert(diff >= 0.0);
        sum_abs_diff = sum_abs_diff + diff;
      }
    }
  }

  if (sum_abs_diff > EPSILON)
  {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
            sum_abs_diff, EPSILON);
  }
  else
  {
    printf("COMMENT: sum of absolute differences (%f)  within acceptable range (%f)\n", sum_abs_diff, EPSILON);
  }
}

/* the slow but correct version of matmul written by David */
void multichannel_conv(float ***image, int16_t ****kernels,
                       float ***output, int width, int height,
                       int nchannels, int nkernels, int kernel_order)
{
  int h, w, x, y, c, m;
  FILE *file_ptr = fopen("output_david.txt","w");

  for (m = 0; m < nkernels; m++)
  {
    for (w = 0; w < width; w++)
    {
      for (h = 0; h < height; h++)
      {
        double sum = 0.0;
        for (c = 0; c < nchannels; c++)
        {
          for (x = 0; x < kernel_order; x++)
          {
            for (y = 0; y < kernel_order; y++)
            {
              sum += image[w + x][h + y][c] * kernels[m][c][x][y];
              // printf("Image:%f Kernel:%d\n",image[w + x][h + y][c], kernels[m][c][x][y]);
            }
          }
          output[m][w][h] = (float)sum;
          if (c == 3) {
            fprintf(file_ptr,"output[%d][%d][%d] = %f\n",m, w, h, (float)sum);
          }
        }
      }
    }
  }
  fclose(file_ptr);
}

double multichannel_conv_insides(float ***image, int16_t ****kernels,
                                 int nchannels, int kernel_order, int w, int h, int m)
{
  double sum = 0.0;
  double lsum = 0.0;
#pragma omp parallel private(lsum)
  {
    int c, x, y;
#pragma omp for collapse(3)
    for (c = 0; c < nchannels; c++)
    {
      for (x = 0; x < kernel_order; x++)
      {
        for (y = 0; y < kernel_order; y++)
        {

          lsum += image[w + x][h + y][c] * kernels[m][c][x][y];
        }
      }
    }
#pragma omp critical
    {
      sum += lsum;
    }
  }
  return sum;
}

void multichannel_conv_inner(float ***image, int16_t ****kernels,
                             float ***output, int width, int height,
                             int nchannels, int nkernels, int kernel_order)
{
  int h, w, x, y, c, m;

  for (m = 0; m < nkernels; m++)
  {
    for (w = 0; w < width; w++)
    {
      for (h = 0; h < height; h++)
      {
        double sum = 0.0;
        double lsum = 0.0;
#pragma parallel for collapse(3)
        for (c = 0; c < nchannels; c++)
        {
          for (x = 0; x < kernel_order; x++)
          {
            for (y = 0; y < kernel_order; y++)
            {
              double tmp = image[w + x][h + y][c] * kernels[m][c][x][y];
#pragma omp critical
              {
                sum += tmp;
              }
            }
          }
          output[m][w][h] = (float)sum;
        }
      }
    }
  }
}

void multichannel_conv_outer(float ***image, int16_t ****kernels,
                             float ***output, int width, int height,
                             int nchannels, int nkernels, int kernel_order)
{
  int h, w, x, y, c, m;
  int channel_width = nchannels - nchannels % 4;
  FILE *file_ptr = fopen("output_student.txt","a");

// #pragma omp parallel for private (h, w, x, y, c, m) shared (image, kernels, output)
  for (m = 0; m < nkernels; m++)
  {
    for (w = 0; w < width; w++)
    {
      for (h = 0; h < height; h++)
      {
        __m128 sum_vector = _mm_setzero_ps();
        __m128 image_channels = _mm_setzero_ps();
        __m128 kernel_channels = _mm_setzero_ps();
        float temp_sum[4];
        for (c = 0; c < channel_width; c += 4)
        {
	      __m128 image_channels = _mm_setzero_ps();
	      __m128 kernel_channels = _mm_setzero_ps();
          for (x = 0; x < kernel_order; x++)
          {
            for (y = 0; y < kernel_order; y++)
            {
              //sum += image[w + x][h + y][c] * kernels[m][c][x][y];
              image_channels = _mm_set_ps(image[w + x][h + y][c+3], image[w + x][h + y][c+2], image[w + x][h + y][c+1], image[w + x][h + y][c]);
              kernel_channels = _mm_set_ps((float)kernels[m][c+3][x][y], (float)kernels[m][c+2][x][y], (float)kernels[m][c+1][x][y], (float)kernels[m][c][x][y]); 
	            sum_vector = _mm_add_ps(sum_vector, _mm_mul_ps(kernel_channels,image_channels));
            }
          }

          float temp[4];
          _mm_storeu_ps(temp, sum_vector);
          double sum = temp[0]+temp[1]+temp[2]+temp[3];
    
          //Deal with the leftovers
          for (c = channel_width; c < nchannels; c++) {
            for (x = 0; x < kernel_order; x++) {
              for (y = 0; y < kernel_order; y++) {
                sum += image[w + x][h + y][c] * kernels[m][c][x][y];
              }
            }
          }

          output[m][w][h] = (float)sum;
          fprintf(file_ptr,"output[%d][%d][%d] = %f\n",m, w, h, (float)sum);
        }
      }
    }
  }
  fclose(file_ptr);
}

// void inner_sse()
// {
//   double sum = 0.0;
//   for (c = 0; c < nchannels; c++)
//   {
//     for (x = 0; x < kernel_order; x++)
//     {
//       for (y = 0; y < kernel_order; y++)
//       {
//         sum += image[w + x][h + y][c] * kernels[m][c][x][y];
//       }
//     }
//     output[m][w][h] = (float)sum;
//   }
// }

int maxim(int a, int b)
{
  return a > b ? a : b;
}

int pick_algo(int width, int height, int nchannels, int nkernels,
              int kernel_order)
{
  int max = MIN;
  const int wkh = width * height * nkernels;
  const int channels = nchannels;
  max = maxim(wkh, max);
  max = maxim(nchannels, max);

  if (max == MIN)
  {
    return 0;
  }
  else if (max == wkh)
  {
    return 1;
  }
  if (max == nchannels)
  {
    return 2;
  }
  return 0;
}

/* the fast version of matmul written by the student */
void student_conv(float ***image, int16_t ****kernels, float ***output,
                  int width, int height, int nchannels, int nkernels,
                  int kernel_order)
{
  int val = pick_algo(width, height, nchannels, nkernels, kernel_order);
  printf("outer\noption: %d \n", val);

  // this call here is just dummy code that calls the slow, simple, correct version.
  // insert your own code instead
  multichannel_conv_outer(image, kernels, output, width,
                          height, nchannels, nkernels, kernel_order);
}

int main(int argc, char **argv)
{
  // float image[W][H][C];
  // float kernels[M][C][K][K];
  // float output[M][W][H];

  float ***image;
  int16_t ****kernels;
  float ***control_output, ***output;
  long long mul_time;
  long long mul_time_david;
  int width, height, kernel_order, nchannels, nkernels;
  struct timeval start_time;
  struct timeval stop_time;
  struct timeval start_time_david;
  struct timeval stop_time_david;

  if (argc != 6)
  {
    fprintf(stderr, "Usage: conv-harness <image_width> <image_height> <kernel_order> <number of channels> <number of kernels>\n");
    exit(1);
  }
  else
  {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    kernel_order = atoi(argv[3]);
    nchannels = atoi(argv[4]);
    nkernels = atoi(argv[5]);
  }
  switch (kernel_order)
  {
  case 1:
  case 3:
  case 5:
  case 7:
    break;
  default:
    fprintf(stderr, "FATAL: kernel_order must be 1, 3, 5 or 7, not %d\n",
            kernel_order);
    exit(1);
  }

  /* allocate the matrices */
  image = gen_random_3d_matrix_float(width + kernel_order, height + kernel_order,
                                     nchannels);
  kernels = gen_random_4d_matrix_int16(nkernels, nchannels, kernel_order, kernel_order);
  output = new_empty_3d_matrix_float(nkernels, width, height);
  control_output = new_empty_3d_matrix_float(nkernels, width, height);

  // DEBUGGING(write_out(A, a_dim1, a_dim2));

  gettimeofday(&start_time_david, NULL);
  /* use a simple multichannel convolution routine to produce control result */
  multichannel_conv(image, kernels, control_output, width,
                    height, nchannels, nkernels, kernel_order);

  gettimeofday(&stop_time_david, NULL);

  mul_time_david = (stop_time_david.tv_sec - start_time_david.tv_sec) * 1000000L +
             (stop_time_david.tv_usec - start_time_david.tv_usec);
             
  printf("David conv time: %lld microseconds\n", mul_time_david);

  /* record starting time of student's code*/
  gettimeofday(&start_time, NULL);

  /* perform student's multichannel convolution */
  student_conv(image, kernels, output, width,
               height, nchannels, nkernels, kernel_order);

  /* record finishing time */
  gettimeofday(&stop_time, NULL);
  mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
             (stop_time.tv_usec - start_time.tv_usec);
  printf("Student conv time: %lld microseconds\n", mul_time);

  DEBUGGING(write_out(output, nkernels, width, height));

  /* now check that the student's multichannel convolution routine
     gives the same answer as the known working version */
  check_result(output, control_output, nkernels, width, height);

  return 0;
}
