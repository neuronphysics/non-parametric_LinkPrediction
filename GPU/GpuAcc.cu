#include "GpuAcc.h"

using namespace std;

#define blocksize 8
#define TILE_DIM 16

__global__ void nodiag_normalize(double *A, double *I, int n, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        if (x == i && x != y) {
            I[x * n + y] /= A[i * n + i];
            A[x * n + y] /= A[i * n + i];
        }

}

__global__ void diag_normalize(double *A, double *I, int n, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        if (x == y && x == i) {
            I[x * n + y] /= A[i * n + i];
            A[x * n + y] /= A[i * n + i];
        }

}

__global__ void gaussjordan(double *A, double *I, int n, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        if (x != i) {
            I[x * n + y] -= I[i * n + y] * A[x * n + i];
            if (y != i) {
                A[x * n + y] -= A[i * n + y] * A[x * n + i];
            }
        }
    }

}

__global__ void set_zero(double *A, double *I, int n, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        if (x != i) {
            if (y == i) {
                A[x * n + y] = 0;
            }
        }
    }
}


void setUpCUDA(double *L, double *iL, int matSize) {

    double *d_A, *d_L, *I, *dI;
    float time;
    cudaError_t err;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int ddsize = matSize * matSize * sizeof(double);

    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 numBlocks((matSize + blocksize - 1) / blocksize, (matSize + blocksize - 1) / blocksize);

    // memory allocation
    err = cudaMalloc((void **) &d_A, ddsize);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMalloc((void **) &dI, ddsize);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    I = new double[matSize * matSize];

    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            if (i == j) I[i * matSize + i] = 1.0;
            else I[i * matSize + j] = 0.0;
        }
    }

    //copy data from CPU to GPU
    err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    //timer start
    cudaEventRecord(start, 0);

    // L^(-1)
    for (int i = 0; i < matSize; i++) {
        nodiag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
        diag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
        gaussjordan <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
        set_zero <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //copy data from GPU to CPU
    err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

//    cout << "Cuda Time - inverse: " << time << "ms\n";

    cudaFree(d_A);
    cudaFree(dI);


    delete[]I;
}

void gpuInverseMethod1(gsl_matrix *original, gsl_matrix *inverseM) {
    size_t matrixSize = original->size1;

    auto *in = new double[matrixSize * matrixSize];
    auto *out = new double[matrixSize * matrixSize];

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            in[i * matrixSize + j] = gsl_matrix_get(original, i, j);
        }
    }

    setUpCUDA(in, out, matrixSize);

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            gsl_matrix_set(inverseM, i, j, out[i * matrixSize + j]);
        }
    }

    delete[] in;
    delete[] out;
}


__global__ void
MatMulNoShared(double *A, double *B, double *C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols,
               double scale1, double scale2) {

    double CValue = 0;

    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {

        for (int n = 0; n < TILE_DIM; ++n)
            if ((k * TILE_DIM + n < ACols && Row < ARows) && (k * TILE_DIM + n < BRows && Col < BCols))
                CValue += A[Row * ACols + k * TILE_DIM + n] * B[(k * TILE_DIM + n) * BCols + Col];

    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] =
                scale1 * CValue +
                scale2 * C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x];
}

void multiplyAndPlus(int CRows, int ACols, int BCols, double scale1, double scale2, double *A, double *B, double *C) {

    int CCols = BCols, ARows = CRows, BRows = ACols;
    cudaError_t err;

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (CCols + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (CRows + dimBlock.y - 1) / dimBlock.y;

    double *deviceA, *deviceB, *deviceC;

    err = cudaMalloc((void **) &deviceA, ARows * ACols * sizeof(double));
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMalloc((void **) &deviceB, BRows * BCols * sizeof(double));
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMalloc((void **) &deviceC, CRows * CCols * sizeof(double));
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    err = cudaMemcpy(deviceA, A, ARows * ACols * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(deviceB, B, BRows * BCols * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(deviceC, C, CRows * CCols * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    MatMulNoShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, ARows, ACols, BRows, BCols, CRows, CCols, scale1,
                                          scale2);

    err = cudaMemcpy(C, deviceC, CRows * CCols * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}


void parseGslMatrix(double *in, const gsl_matrix *original, CBLAS_TRANSPOSE_t trans) {
    if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
        for (int i = 0; i < original->size1; i++) {
            for (int j = 0; j < original->size2; j++) {
                in[i * original->size2 + j] = gsl_matrix_get(original, i, j);
            }
        }
    } else {
        for (int i = 0; i < original->size1; i++) {
            for (int j = 0; j < original->size2; j++) {
                in[j * original->size1 + i] = gsl_matrix_get(original, i, j);
            }
        }
    }
}


void
gpuMatrixMultiply(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C, double scale1, double scale2,
                  CBLAS_TRANSPOSE_t transA,
                  CBLAS_TRANSPOSE_t transB) {
    auto *inputA = new double[A->size1 * A->size2];
    auto *inputB = new double[B->size1 * B->size2];
    auto *inputC = new double[C->size1 * C->size2];

    parseGslMatrix(inputA, A, transA);
    parseGslMatrix(inputB, B, transB);
    parseGslMatrix(inputC, C, CBLAS_TRANSPOSE::CblasNoTrans);

    int ACols = A->size2;
    if (transA == CBLAS_TRANSPOSE::CblasTrans) {
        ACols = A->size1;
    }

    int BCols = B->size2;
    if (transB == CBLAS_TRANSPOSE::CblasTrans) {
        BCols = B->size1;
    }
    multiplyAndPlus(C->size1, ACols, BCols, scale1, scale2, inputA, inputB, inputC);

    for (int i = 0; i < C->size1; i++) {
        for (int j = 0; j < C->size2; j++) {
            gsl_matrix_set(C, i, j, inputC[i * C->size2 + j]);
        }
    }

    delete[] inputA;
    delete[] inputB;
    delete[] inputC;
}


__global__ void
KronDevice(double *A, double *B, double *out, int Arow, int Acol, int Brow, int Bcol, int Rrow, int Rcol) {
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    if ((Row < Arow) && (Col < Acol)){
        double factor = A[Row * Acol + Col];
        int outR = Row * Brow;
        int outC = Col * Bcol;
        for(int i = 0; i < Brow; i++){
            for(int j = 0; j < Bcol; j++){
                out[(outR + i) * Rcol + outC + j] = factor * B[i * Bcol + j];
            }
        }
    }

}


void kron(double *A, double *B, double *out, int Arow, int Acol, int Brow, int Bcol) {

    int Asize = Arow * Acol;
    int Bsize = Brow * Bcol;
    cudaError_t err;

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (Acol + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (Arow + dimBlock.y - 1) / dimBlock.y;

    double *deviceA, *deviceB, *deviceC;

    err = cudaMalloc((void **) &deviceA, Asize * sizeof(double));
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMalloc((void **) &deviceB, Bsize * sizeof(double));
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMalloc((void **) &deviceC, Asize * Bsize * sizeof(double));
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    err = cudaMemcpy(deviceA, A, Asize * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(deviceB, B, Bsize * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(deviceC, out, Asize * Bsize * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    KronDevice<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, Arow, Acol, Brow, Bcol, Arow * Brow, Acol * Bcol);

    err = cudaMemcpy(out, deviceC, Asize * Bsize * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}

void
gpuKron(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *Res) {
    auto *inputA = new double[A->size1 * A->size2];
    auto *inputB = new double[B->size1 * B->size2];
    auto *output = new double[Res->size1 * Res->size2];


    parseGslMatrix(inputA, A, CBLAS_TRANSPOSE::CblasNoTrans);
    parseGslMatrix(inputB, B, CBLAS_TRANSPOSE::CblasNoTrans);


    kron(inputA, inputB, output, (int) (A->size1), (int) (A->size2), (int) (B->size1), int(B->size2));

    for (int i = 0; i < Res->size1; i++) {
        for (int j = 0; j < Res->size2; j++) {
            gsl_matrix_set(Res, i, j, output[i * Res->size2 + j]);
        }
    }

    delete[] inputA;
    delete[] inputB;
    delete[] output;
}