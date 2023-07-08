#include <cusolverDn.h>
#include "GpuAcc.h"

using namespace std;

#define blocksize 8
#define TILE_DIM 16
#define BLOCK_DIM 16

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

    // L^(-1)
    for (int i = 0; i < matSize; i++) {
        nodiag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
        diag_normalize <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
        gaussjordan <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
        set_zero <<< numBlocks, threadsPerBlock >>>(d_A, dI, matSize, i);
    }

    //copy data from GPU to CPU
    err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }


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

    if ((Row < Arow) && (Col < Acol)) {
        double factor = A[Row * Acol + Col];
        int outR = Row * Brow;
        int outC = Col * Bcol;
        for (int i = 0; i < Brow; i++) {
            for (int j = 0; j < Bcol; j++) {
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

// wait until cusolver could be used
/**
void symmetricAndPDMatrixInverse(gsl_matrix *matrix) {
    int N = matrix->size1;

    // identity matrix
    auto *h_I = new double[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                h_I[i * N + j] = 1;
            } else {
                h_I[i * N + j] = 0;
            }
        }
    }

    double *d_I;
    cudaMalloc(&d_I, N * N * sizeof(double));

    // Move the relevant matrix from host to device
    cudaMemcpy(d_I, h_I, N * N * sizeof(double), cudaMemcpyHostToDevice);


    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);

    // --- CUBLAS initialization
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);


    // Setting the host, N x N matrix
    auto *h_A = new double[N * N];
    parseGslMatrix(h_A, matrix, CblasNoTrans);

    // Allocate device space for the input matrix
    double *d_A;
    cudaMalloc(&d_A, N * N * sizeof(double));

    // Move the relevant matrix from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);



    // COMPUTING THE CHOLESKY DECOMPOSITION

    // --- cuSOLVE input/output parameters/arrays
    int work_size = 0;
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));

    // --- CUDA CHOLESKY initialization
    cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &work_size);

    // --- CUDA POTRF execution
    double *work;
    cudaMalloc(&work, work_size * sizeof(double));

    cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, work, work_size, devInfo);
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (devInfo_h != 0) cout << "Unsuccessful potrf execution\n\n" << "devInfo = " << devInfo_h << "\n\n";

    // --- At this point, the lower triangular of matrix is stored in d_A, and it is a unit triangular matrix.
    const double alpha = 1.f;
    const double beta = 0;
    // solve L ^ -1
    cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N,
                &alpha, d_A, N, d_I, N);

    // by here d_I contain L^-1, copy it to d_A
    cudaMemcpy(d_A, d_I, N * N * sizeof(double), cudaMemcpyDeviceToDevice);


    // alloc device space for final result
    double *d_R;
    cudaMalloc(&d_R, N * N * sizeof(double));

    // perform d_A^T * d_I = d_R
    cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_I, N, &beta, d_R, N);

    // by here d_R contains matrix inverse
    cudaMemcpy(h_A, d_R, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            gsl_matrix_set(matrix, i, j, h_A[i * N + j]);
        }
    }

    cusolverDnDestroy(solver_handle);
    cublasDestroy(cublas_handle);
    delete[] h_I;
    delete[] h_A;
    cudaFree(d_A);
    cudaFree(d_I);
    cudaFree(d_R);

    cudaFree(work);
    cudaFree(devInfo);
}
 */

__global__ void transpose(double *dest, double *source, int width, int height)
{
    __shared__ double block[BLOCK_DIM][BLOCK_DIM+1];

    // read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = source[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();

    // write the transposed matrix tile to global memory (odata) in linear order
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        dest[index_out] = block[threadIdx.x][threadIdx.y];
    }
}


void gpuBoostedComputeFullEta(const gsl_matrix *Z, const gsl_matrix *Rho, gsl_matrix *etaKK) {
    int K = Z->size1;
    int N = Z->size2;

    // prepare Z
    auto *h_Z = new double[K * N];
    parseGslMatrix(h_Z, Z, CblasNoTrans);
    double *d_Z;
    cudaMalloc(&d_Z, K * N * sizeof(double));
    cudaMemcpy(d_Z, h_Z, K * N * sizeof(double), cudaMemcpyHostToDevice);

    // prepare ZT
    double *d_ZT;
    cudaMalloc(&d_ZT, K * N * sizeof(double));
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM, (K + BLOCK_DIM - 1) / BLOCK_DIM, 1);
    dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    transpose<<< grid, threads >>>(d_ZT, d_Z, N, K);


    // prepare Rho
    auto *h_R = new double[N * N];
    parseGslMatrix(h_R, Rho, CblasNoTrans);
    double *d_R;
    cudaMalloc(&d_R, N * N * sizeof(double));
    cudaMemcpy(d_R, h_R, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // prepare intermediate var
    double *d_zR;
    cudaMalloc(&d_zR, K * N * sizeof(double));

    // prepare final res
    auto *h_E = new double[K * K];
    double *d_E;
    cudaMalloc(&d_E, K * K * sizeof(double));


    multiplyAndPlus(K, N, N, 1, 0, d_Z, d_R, d_zR);
    multiplyAndPlus(K, N, K, 1, 0, d_zR, d_ZT, d_E);

    cudaMemcpy(h_E, d_E, K * K * sizeof(double), cudaMemcpyDeviceToHost);


    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            gsl_matrix_set(etaKK, i, j, h_E[i * K + j]);
        }
    }

    delete[] h_Z;
    delete[] h_R;
    delete[] h_E;
    cudaFree(d_Z);
    cudaFree(d_R);
    cudaFree(d_zR);
    cudaFree(d_E);
    cudaFree(d_ZT);
}

void
gpuBoostedEtaUpdate(int N, int K, const double *znkZ, const double *Zkzn, const double *znkzn,
                    const gsl_matrix *rho_col, const gsl_matrix *rho_row,
                    double rho_nn, const gsl_matrix *fullEta, gsl_matrix *etanon) {
    double *d_znkZ;
    cudaMalloc(&d_znkZ, K * K * N * sizeof(double));
    cudaMemcpy(d_znkZ, znkZ, K * K * N * sizeof(double), cudaMemcpyHostToDevice);

    double *d_Zkzn;
    cudaMalloc(&d_Zkzn, K * K * N * sizeof(double));
    cudaMemcpy(d_Zkzn, Zkzn, K * K * N * sizeof(double), cudaMemcpyHostToDevice);

    double *d_znkzn;
    cudaMalloc(&d_znkzn, K * K * sizeof(double));
    cudaMemcpy(d_znkzn, znkzn, K * K * sizeof(double), cudaMemcpyHostToDevice);

    auto *h_rho_col = new double[N];
    parseGslMatrix(h_rho_col, rho_col, CblasNoTrans);
    double *d_rho_col;
    cudaMalloc(&d_rho_col, N * sizeof(double));
    cudaMemcpy(d_rho_col, h_rho_col, N * sizeof(double), cudaMemcpyHostToDevice);

    auto *h_rho_row = new double[N];
    parseGslMatrix(h_rho_row, rho_row, CblasNoTrans);
    double *d_rho_row;
    cudaMalloc(&d_rho_row, N * sizeof(double));
    cudaMemcpy(d_rho_row, h_rho_row, N * sizeof(double), cudaMemcpyHostToDevice);

    double *d_rho_nn;
    cudaMalloc(&d_rho_nn, 1 * sizeof(double));
    cudaMemcpy(d_rho_nn, &rho_nn, 1 * sizeof(double), cudaMemcpyHostToDevice);

    auto *h_full_eta = new double[K * K];
    parseGslMatrix(h_full_eta, fullEta, CblasNoTrans);
    double *d_full_eta;
    cudaMalloc(&d_full_eta, K * K * sizeof(double));
    cudaMemcpy(d_full_eta, h_full_eta, K * K * sizeof(double), cudaMemcpyHostToDevice);


    multiplyAndPlus(K * K, N, 1, -1, 1, d_znkZ, d_rho_row, d_full_eta);
    multiplyAndPlus(K * K, N, 1, -1, 1, d_Zkzn, d_rho_col, d_full_eta);
    multiplyAndPlus(K * K, 1, 1, 1, 1, d_znkzn, d_rho_nn, d_full_eta);

    cudaMemcpy(h_full_eta, d_full_eta, K * K * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < etanon->size1; i++) {
        for (int j = 0; j < etanon->size2; j++) {
            gsl_matrix_set(etanon, i, j, h_full_eta[i * etanon->size2 + j]);
        }
    }

    delete[] h_rho_col;
    delete[] h_rho_row;
    delete[] h_full_eta;
    cudaFree(d_Zkzn);
    cudaFree(d_znkZ);
    cudaFree(d_znkzn);
    cudaFree(d_rho_col);
    cudaFree(d_rho_row);
    cudaFree(d_rho_nn);
    cudaFree(d_full_eta);
}