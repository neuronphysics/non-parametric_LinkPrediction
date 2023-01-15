#include "gpuAcc.h"

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
        nodiag_normalize << < numBlocks, threadsPerBlock >> >(d_A, dI, matSize, i);
        diag_normalize << < numBlocks, threadsPerBlock >> >(d_A, dI, matSize, i);
        gaussjordan << < numBlocks, threadsPerBlock >> >(d_A, dI, matSize, i);
        set_zero << < numBlocks, threadsPerBlock >> >(d_A, dI, matSize, i);
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


void parseGslMatrix(double *in, gsl_matrix *original, CBLAS_TRANSPOSE_t trans) {
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
gpuMatrixMultiply(gsl_matrix *A, gsl_matrix *B, gsl_matrix *C, double scale1, double scale2, CBLAS_TRANSPOSE_t transA,
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


__global__
void Transpose(float *matrix, float *t_matrix, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; //index row
    int j = blockIdx.y * blockDim.y + threadIdx.y; //index col
    int stride = gridDim.x * blockDim.x;

    while (i < N && j < N) {
        while (j < N && i < N) { // loop in case memory is not enough to do it one shot
            t_matrix[i * N + j] = matrix[j * N + i];
            j += stride;
        }
        j = blockIdx.y * blockDim.y + threadIdx.y;
        i += stride;
    }
    return;
}


__global__
void Matrix_Mul(float *matrix_1, float *matrix_2, float *matrix_m, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > N || j > N) return;
    for (int k = 0; k < N; ++k) {
        matrix_m[i * N + j] += (matrix_1[i * N + k]) * (matrix_2[k * N + j]);
    }
}


// Function to get cofactor
void getCofactor(float *A, float *temp, int p, int q, int n, int N) {
    int i = 0, j = 0;

    // Looping for each element of the matrix
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            // Copying into temporary matrix only those element
            // which are not in given row and column
            if (row != p && col != q) {
                temp[i * N + j++] = A[row * N + col];

                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

// Recursive function for finding determinant of matrix.
int determinant(float *A, int n, int N) {
    int D = 0; // Initialize result

    // Base case : if matrix contains single element
    if (n == 1)
        return A[0];

    auto *temp = new float [N * N]; // To store cofactors

    int sign = 1; // To store sign multiplier

    // Iterate for each element of first row
    for (int f = 0; f < n; f++) {
        // Getting Cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n, N);
        D += sign * A[0 * N + f] * determinant(temp, n - 1, N);

        // terms are to be added with alternate sign
        sign = -sign;
    }

    delete[] temp;
    return D;
}

// Function to get adjoint
void adjoint(float *A, float *adj, int N) {
    if (N == 1) {
        adj[0] = 1;
        return;
    }

    // temp is used to store cofactors
    int sign = 1;
    auto *temp = new float [N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Get cofactor
            getCofactor(A, temp, i, j, N, N);

            // sign of adj positive if sum of row
            // and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;

            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            adj[j * N + i] = (sign) * (determinant(temp, N - 1, N));
        }
    }

    delete[] temp;
}

// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(float *A, float *inverse, int N) {
    // Find determinant of A[][]
    int det = determinant(A, N, N);
    if (det == 0) {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }

    // Find adjoint
    float * adj = new float [N * N];
    adjoint(A, adj, N);

    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            inverse[i * N + j] = adj[i * N + j] / float(det);

    delete[] adj;
    return true;
}


void gpuInverseMethod2(gsl_matrix *original, gsl_matrix *inverseM) {
    size_t matrixSize = original->size1;

    auto *in = new float[matrixSize * matrixSize];
    auto *out = new float[matrixSize * matrixSize];

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            in[i * matrixSize + j] = (float) gsl_matrix_get(original, i, j);
        }
    }


    float *t_matrix, *in_matrix, *d_matrix_mult, *d_inv, *d_out;
    cudaMalloc((void **) &t_matrix, matrixSize * matrixSize * sizeof(float ));
    cudaMalloc((void **) &in_matrix, matrixSize * matrixSize * sizeof(float ));
    cudaMalloc((void **) &d_matrix_mult, matrixSize * matrixSize * sizeof(float ));
    cudaMalloc((void **) &d_inv, matrixSize * matrixSize * sizeof(float ));
    cudaMalloc((void **) &d_out, matrixSize * matrixSize * sizeof(float ));



    float *matrix_mult = new float[matrixSize * matrixSize];
    float *adj = new float[matrixSize * matrixSize]; // To store adjoint
    float *inv = new float[matrixSize * matrixSize];

    cudaMemcpy(in_matrix, in, matrixSize * matrixSize, cudaMemcpyHostToDevice);

    Transpose<<<8, 128>>>(in_matrix, t_matrix, matrixSize);


    Matrix_Mul<<<8, 128>>>(t_matrix, in_matrix, d_matrix_mult, matrixSize);


    cudaMemcpy(matrix_mult, d_matrix_mult, matrixSize * matrixSize, cudaMemcpyDeviceToHost);



    inverse(matrix_mult, inv, matrixSize);


    cudaMemcpy(d_inv, inv, matrixSize * matrixSize, cudaMemcpyHostToDevice);


    Matrix_Mul<<<8, 128>>>(d_inv, t_matrix, d_out, matrixSize);


    cudaMemcpy(out, d_out, matrixSize * matrixSize, cudaMemcpyDeviceToHost);


    delete[] matrix_mult;
    delete[] adj;
    delete[] inv;

    cudaFree(t_matrix);
    cudaFree(in_matrix);
    cudaFree(d_matrix_mult);
    cudaFree(d_inv);
    cudaFree(d_out);

    for (int i = 0; i < matrixSize; i++) {
        for (int j = 0; j < matrixSize; j++) {
            gsl_matrix_set(inverseM, i, j, out[i * matrixSize + j]);
        }
    }

    delete[] in;
    delete[] out;
}





