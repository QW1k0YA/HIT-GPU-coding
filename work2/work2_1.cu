#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "vector"
#define N 6400
using namespace std;

__global__ void muiltiply(double *A,double *B,double *C,int width) {
    int x_index = threadIdx.x + blockDim.x * blockIdx.x;
    int y_index = threadIdx.y + blockDim.y * blockIdx.y;

    if (x_index < width && y_index < width) {
        int sum = 0;
        for (int i = 0; i < width; i++) {
            sum += A[y_index * width + i] * B[i * width + x_index];
        }
        C[x_index + y_index * width] = sum;
    }

}

int main()
{
    int n =N;
    double *dev_A,*dev_B,*dev_C;
    double A[N],B[N],C[N];

    for(int i = 0;i < n;i ++)
    {
        A[i] = i;
        B[i] = i*i%8;
    }

    cudaMalloc(&dev_A, n * sizeof(double));
    cudaMalloc(&dev_B, n * sizeof(double));
    cudaMalloc(&dev_C, n * sizeof(double));

    cudaMemcpy(dev_A,A,n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B,B,n*sizeof(double),cudaMemcpyHostToDevice);

    ///<<<numBlocks, blockSize>>>
    ///choose the size
    dim3 blockdim(8,8);
    dim3 griddim(10,10);
    double t1,t2;
    t1 = clock();
    muiltiply<<<griddim,blockdim>>>(dev_A,dev_B,dev_C, sqrt(n));
    t2 = clock();


    cudaMemcpy(C,dev_C,n*sizeof(double),cudaMemcpyDeviceToHost);

    cout << "A:\n";
    int width = sqrt(n);

    for(int ii = 0; ii < n;ii += width)
    {
        for(int i = ii;i < ii + width;i ++)
        {
            cout << A[i] << " ";
        }
        cout << "\n";
    }


    cout << "\nB:\n";
    for(int ii = 0; ii < n;ii += width)
    {
        for(int i = ii;i < ii + width;i ++)
        {
            cout << B[i] << " ";
        }
        cout << "\n";
    }

    cout << "\nC:\n";
    for(int ii = 0; ii < n;ii += width)
    {
        for(int i = ii;i < ii + width;i ++)
        {
            cout << C[i] << " ";
        }
        cout << "\n";
    }

    cout <<(t2 - t1)/CLOCKS_PER_SEC<<"secs" << endl;
}

