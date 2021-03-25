//#include <iostream>
//#include <math.h>
//using namespace std;
//// Kernel function to add the elements of two arrays
//__global__
//void add(int n, float *x, float *y)
//{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//    for (int i = index; i < n; i += stride)
//        y[i] = x[i] + y[i];
//}
//
//int main(void)
//{
//    int N = 1;
//    N = N << 31;
//    float *x, *y;
//
//    // Allocate Unified Memory – accessible from CPU or GPU
//    cudaMallocManaged(&x, N*sizeof(float));
//    cudaMallocManaged(&y, N*sizeof(float));
//
//    // initialize x and y arrays on the host
//    for (int i = 0; i < N; i++) {
//        x[i] = 0.00000001f;
//        y[i] = 0.00000003f;
//    }
//
//    for(int temp = 0; temp<1000000000; temp++)
//    {
////        cout << temp;
//        // Run kernel on 1M elements on the GPU
//
//        int blockSize = 4096*8;
//        int numBlocks = (N + blockSize - 1) / blockSize;
//        add<<<numBlocks, blockSize>>>(N, x, y);
//
//        // Wait for GPU to finish before accessing on host
//        cudaDeviceSynchronize();
//        if (temp% 100000 == 9999){
//            cout << temp << "\n";
//            cout << y[0] << "\n";
//        }
//    }
//
//
//    // Check for errors (all values should be 3.0f)
//    float maxError = 0.0f;
//    for (int i = 0; i < N; i++)
//        maxError = fmax(maxError, fabs(y[i]-3.0f));
//    std::cout << "Max error: " << maxError << std::endl;
//
//    // Free memory
//    cudaFree(x);
//    cudaFree(y);
//
//    return 0;
//}