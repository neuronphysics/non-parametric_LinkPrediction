#include <iomanip>
#include <thread>
#include "Utils.h"


using namespace std;

ofstream matrixOut;
uint64_t timeSeed;


double fre_1(double x, double func, double mu, double w) {
    return w * (x - mu);
}

double f_1(double x, double func, double mu, double w) {

    if (func == 1) {
        return log(gsl_sf_exp(w * (x - mu)) - 1);
    } else if (func == 2) {
        return sqrt(w * (x - mu));
    } else {
        LOG(OUTPUT_NORMAL, "error: unknown transformation function. Used default transformation log(exp(y)-1)")
        return log(gsl_sf_exp(w * (x - mu)) - 1);
    }
}

double f_w(double x, double func, double mu, double w) {
    if (func == 1) {
        if (x != 0) {
            return (1 / w) * (log(gsl_sf_exp(x) - 1) - mu);
        } else {
            return 0;
        }
    } else if (func == 2) {
        return sqrt(w * (x - mu));
    } else {
        LOG(OUTPUT_NORMAL, "error: unknown transformation function. Used default transformation log(exp(y)-1)")
        if (x != 0) {
            return (1 / w) * (log(gsl_sf_exp(x) - 1) - mu);
        } else {
            return 0;
        }
    }
}


/**
 * Compute the mean of the input vector exclude missing entries
 * @param N Input, length of vector
 * @param missing  Input, symbol of the missing entries
 * @param v Input, vector to compute
 * @return mean of the vector
 */
double compute_vector_mean(int N, double missing, const gsl_vector *v) {
    double sumX = 0;
    double countX = 0;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing && !(gsl_isnan(xnd))) {
            sumX += xnd;
            countX += 1;
        }
    }
    return sumX / countX;
}

/**
 * Compute the variance of the input vector exclude missing entries
 * @param N Input, length of vector
 * @param missing  Input, symbol of the missing entries
 * @param v Input, vector to compute
 * @return variance of the vector
 */
double compute_vector_var(int N, double missing, const gsl_vector *v) {
    double meanX = compute_vector_mean(N, missing, v);
    double sumX = 0;
    double countX = 0;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing && !(gsl_isnan(xnd))) {
            sumX += pow(xnd - meanX, 2);
            countX += 1;
        }
    }
    return sumX / countX;
}

/**
 * Find the max value of the input vector exclude missing entries
 * @param N Input, length of vector
 * @param missing  Input, symbol of the missing entries
 * @param v Input, vector to compute
 * @return max value of the vector
 */
double compute_vector_max(int N, double missing, const gsl_vector *v) {
    double maxX = -1e100;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing) {
            if (xnd > maxX) { maxX = xnd; }
        }
    }
    return maxX;
}


/**
 * Find the min value of the input vector exclude missing entries
 * @param N Input, length of vector
 * @param missing  Input, symbol of the missing entries
 * @param v Input, vector to compute
 * @return min value of the vector
 */
double compute_vector_min(int N, double missing, const gsl_vector *v) {
    double minX = 1e100;
    for (int nn = 0; nn < N; nn++) {
        double xnd = gsl_vector_get(v, nn);
        if (xnd != missing) {
            if (xnd < minX) { minX = xnd; }
        }
    }
    return minX;
}


/**
 * Compute Exp(x) base e, with -300 < x < 300
 * @param x Input, number to take exponential function
 * @return exp(x) for 300 < x < 300 and INF for x > 300, 0 for x < -300
 */
double expFun(double x) {
    if (x > 300) {
        return GSL_POSINF;
    } else if (x < -300) {
        return 0;
    } else {
        return gsl_sf_exp(x);
    }
}

/**
 * Compute the factorial of N
 * @param N Input, number to compute factorial
 * @return N!
 */
int factorial(int N) {
    int fact = 1;
    for (int c = 1; c <= N; c++) {
        fact = fact * c;
    }
    return fact;
}

/**
 * Compute C = alpha * AB + beta * C where A, B, C are matrix
 * @param A Input, matrix factor 1
 * @param B Input, matrix factor 2
 * @param C Input/Output, take the result of computation
 * @param alpha Input, Coefficient of A * B
 * @param beta Input, Coefficient of C
 * @param TransA Input, defines if A needs to take transpose before operation
 * @param TransB Input, defines if B needs to take transpose before operation
 */
void matrix_multiply(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C, double alpha, double beta,
                     CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB) {
    // C = alpha * AB + beta * C
    gpuMatrixMultiply(A, B, C, alpha, beta, TransA, TransB);
}

/**
 * compute ln(abs(det(matrix)))
 * @param matrix Input, matrix to be find determinant
 * @param rows Input, number of rows of input matrix
 * @param cols Input, number of columns of input matrix
 * @return ln(abs(det(matrix)))
 */
double lndet_get(const gsl_matrix *matrix, int rows, int cols) {
    double det;
    int signum;
    gsl_permutation *p = gsl_permutation_alloc(rows);

    gsl_matrix *tmpA = gsl_matrix_calloc(rows, cols);
    gsl_matrix_memcpy(tmpA, matrix);

    gsl_linalg_LU_decomp(tmpA, p, &signum);
    det = gsl_linalg_LU_lndet(tmpA);
    gsl_permutation_free(p);

    gsl_matrix_free(tmpA);

    return det;
}

/**
 * In place compute the inverse of the input matrix
 * @param matrix Input/Output, matrix to find inverse with
 */
void inverse(gsl_matrix *matrix) {
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    gpuInverseMethod1(matrix, matrix);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    LOG(OUTPUT_CONCURRENT, "Inverse cost = %lld [ms]", chrono::duration_cast<chrono::milliseconds>(end - begin).count())
}


//
/**
 * Sampling function, determine if a new feature is needed.
 *      k will not increase unless p[0] is small
 * @param p Input, p[0] contains the major possibility, p[1] contains the complement such that p[0] + p[1] = 1
 * @param nK Input, uncleared meaning for now
 * @return the number of new features needs to be added, in most cases k < 2
 */
int mnrnd(double *p, int nK) {
    double pMin = 0;
    double pMax = p[0];
    double s = rand01();
    int k = 0;
    int flag = 1;
    int Knew;
    while (flag) {
        if ((s > pMin) && (s <= pMax)) {
            flag = 0;
            Knew = k;
        } else {
            pMin += p[k];
            pMax += p[k + 1];
        }
        k++;
    }
    return Knew;
}

/**
 * Sampling function, compute v = decomposition(matrix) * normal distribution sampling + mu
 * @param v Output, v = decomposition(matrix) * normal distribution sampling + mu
 * @param matrix Input, copied and perform cholesky decomposition into lower matrix
 * @param mu Input, mean vector of future v
 * @param K Input, matrix size
 * @param seed Input, random number generator seed
 */
void mvnrnd(gsl_vector *v, gsl_matrix *matrix, gsl_vector *mu, int K, const gsl_rng *seed) {
    gsl_matrix *matrixCopy = gsl_matrix_calloc(K, K);
    gsl_matrix_memcpy(matrixCopy, matrix);

    // decompose matrix_copy to LL^T = matrix, lower matrix
    gsl_linalg_cholesky_decomp(matrixCopy);

    // set vector to random gaussian distribution value
    for (int k = 0; k < K; k++) {
        gsl_vector_set(v, k, gsl_ran_ugaussian(seed));
    }
    // v = matrixCopy(include diagonal) * v
    gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, matrixCopy, v);

    // v = v + Mu
    gsl_vector_add(v, mu);

    // H = decomposition(Q) * normal distribution + Q * Eta
    gsl_matrix_free(matrixCopy);
}

/**
 * Sample from truncated normal distribution on range xlo to xhi
 *      Note that this may fail when truncate at the tail of normal distribution
 * @param mu Input, mean
 * @param sigma Input, variance
 * @param xlo Input, lower bound for sampling
 * @param xhi Input, upper bound for sampling
 * @param rng Input, random number generator seed
 * @return
 */
double truncnormrnd(double mu, double sigma, double xlo, double xhi, const gsl_rng *rng) {
    if (xlo > xhi) {
        LOG(OUTPUT_NORMAL, "error: xlo<xhi")
    }

    // when (xlo - mu) / sigma greater than 5, the result will be 1, resulting z = inf
    double plo = gsl_cdf_gaussian_P((xlo - mu) / sigma, 1.0);
    if (plo >= 1) {
        LOG(OUTPUT_NORMAL, "plo too large, mu = %f, sigma = %f", mu, sigma)
        plo = 0.99999;
    }
    double phi = gsl_cdf_gaussian_P((xhi - mu) / sigma, 1.0);
    if (phi <= 0) {
        LOG(OUTPUT_NORMAL, "phi too small, mu = %f, sigma = %f", mu, sigma)
        phi = 0.00001;
    }

    double r = gsl_rng_uniform(rng);
    double res = r * (phi - plo) + plo;
    if (res >= 1) {
        LOG(OUTPUT_NORMAL, "res too large, mu = %f, sigma = %f, r = %f", mu, sigma, r)
        res = 0.99999999;
    }
    if (res <= 0) {
        LOG(OUTPUT_NORMAL, "res too small, mu = %f, sigma = %f, r = %f", mu, sigma, r)
        res = 0.00000001;
    }

    double z = gsl_cdf_gaussian_Pinv(res, 1.0);
    return mu + z * sigma;
}

/**
 * Helper function for parallelized kronecker product computation
 * @param prod Output, stores the result of A kron B
 * @param a Input, matrix A
 * @param b Input, matrix B
 * @param startJ Input, start index assigned to this thread
 * @param endJ Input, end index assigned to this thread
 */
void kron_parallel(gsl_matrix *prod, const gsl_matrix *a, const gsl_matrix *b, size_t startJ, size_t endJ) {
    for (unsigned int i = 0; i < a->size1; ++i) {
        for (unsigned int j = startJ; j < endJ; ++j) {
            gsl_matrix toto = gsl_matrix_submatrix(prod, i * b->size1, j * b->size2, b->size1, b->size2).matrix;
            gsl_matrix_memcpy(&toto, b);
            gsl_matrix_scale(&toto, a->data[i * a->tda + j]);
        }
    }
}

/**
 * Compute the Kronecker product of A and B
 * @param prod Output, stores the result of A Kron B
 * @param a Input, matrix A
 * @param b Input, matrix B
 */
void gsl_Kronecker_product(gsl_matrix *prod, const gsl_matrix *a, const gsl_matrix *b) {
    if (a->size2 > 600 || b->size2 > 600) {
        thread t1(kron_parallel, prod, a, b, 0, a->size2 / 3);
        thread t2(kron_parallel, prod, a, b, a->size2 / 3, a->size2 / 3 * 2);
        thread t3(kron_parallel, prod, a, b, a->size2 / 3 * 2, a->size2);
        t1.join();
        t2.join();
        t3.join();
    } else {
        kron_parallel(prod, a, b, 0, a->size2);
    }
}

/**
 * Convert matrix to vector
 * @param vect Output, converted matrix
 * @param matrix Input, matrix to be convert
 */
void gsl_matrix2vector(gsl_matrix *vect, gsl_matrix *matrix) {
    for (unsigned int i = 0; i < matrix->size1; ++i) {
        for (unsigned int j = 0; j < matrix->size2; ++j) {
            vect->data[(i * matrix->size2 + j) * vect->tda] = matrix->data[i * matrix->tda + j];
        }
    }
}

/**
 * Convert vector to matrix
 * @param vect Input, vector to convert
 * @param matrix Output, result matrix
 */
void gsl_vector2matrix(gsl_matrix *vect, gsl_matrix *matrix) {
    for (unsigned int i = 0; i < matrix->size1; ++i) {
        for (unsigned int j = 0; j < matrix->size2; ++j) {
            matrix->data[i * matrix->tda + j] = vect->data[(i * matrix->size2 + j) * vect->tda];
        }
    }
}

/**
 * Remove a column from <in> matrix and stores the result in <out> matrix
 * @param K Input, number of rows in <in> matrix
 * @param N Input, number of columns in <in> matrix
 * @param i Input, column index to remove
 * @param out Output, matrix after removing the target column
 * @param in Input, original matrix
 */
void remove_col(int K, int N, int i, gsl_matrix *out, gsl_matrix *in) {
    gsl_matrix_view In_view;
    gsl_matrix_view Out_view;
    if (i == 0) {
        In_view = gsl_matrix_submatrix(in, 0, 1, K, N - 1);
        gsl_matrix_memcpy(out, &In_view.matrix);
    } else if (i == (N - 1)) {
        In_view = gsl_matrix_submatrix(in, 0, 0, K, N - 1);
        gsl_matrix_memcpy(out, &In_view.matrix);
    } else {
        int j = i + 1;
        In_view = gsl_matrix_submatrix(in, 0, 0, K, j);
        Out_view = gsl_matrix_submatrix(out, 0, 0, K, j);
        gsl_matrix_memcpy(&Out_view.matrix, &In_view.matrix);

        In_view = gsl_matrix_submatrix(in, 0, j, K, N - j);
        Out_view = gsl_matrix_submatrix(out, 0, j - 1, K, N - j);
        gsl_matrix_memcpy(&Out_view.matrix, &In_view.matrix);
    }
}

/**
 * Compute the updated Q matrix with new Z matrix
 *      Noted that the original Q matrix will not be involved in the computation
 * @param N Input, number of columns in Z matrix
 * @param K Input, number of rows in Z matrix
 * @param Z Input, the updated Z matrix
 * @param beta Input, coefficient to identity matrix
 * @param Q Output, stores the new Q matrix
 */
void compute_inverse_Q_directly(int N, int K, const gsl_matrix *Z, double beta, gsl_matrix *Q) {
    gsl_matrix *identity = gsl_matrix_calloc(K * K, K * K);
    gsl_matrix_set_identity(identity);
    gsl_matrix_scale(identity, beta);

    gsl_matrix *zzT = gsl_matrix_calloc(K, K);
    matrix_multiply(Z, Z, zzT, 1, 0, CblasNoTrans, CblasTrans);

    gsl_Kronecker_product(Q, zzT, zzT);
    gsl_matrix_add(Q, identity);

    // Q = zz^T Kron zz^T + beta * I
    gsl_matrix_free(identity);
    gsl_matrix_free(zzT);

    inverse(Q);
//    symmetricAndPDMatrixInverse(Q);
}

/**
 * Compute Eta matrix without Zn column in Z matrix
 * @param Znon Input, Z matrix without nth column
 * @param Rho Input, Rho matrix, will be used to compute Rho without nth row and column
 * @param n Input, the index of the Zn column that is removed from Z matrix
 * @param Enon Output, stores the output of (Znon Kron Znon) * vec(Rho_n_n)
 */
void normal_update_eta(const gsl_matrix *Znon, const gsl_matrix *Rho, int n, gsl_matrix *Enon) {
    gsl_matrix *ZnonOZnon = gsl_matrix_calloc(Znon->size1 * Znon->size1, Znon->size2 * Znon->size2);
    gsl_Kronecker_product(ZnonOZnon, Znon, Znon);
    gsl_matrix *rhocy = gsl_matrix_calloc(Rho->size1, Rho->size2);
    gsl_matrix_memcpy(rhocy, Rho);

    for (int i = n; i < Rho->size1 - 1; i++) {
        gsl_matrix_swap_rows(rhocy, i, i + 1);
        gsl_matrix_swap_columns(rhocy, i, i + 1);
    }
    gsl_matrix_view rho_n_n = gsl_matrix_submatrix(rhocy, 0, 0, Rho->size1 - 1, Rho->size2 - 1);
    gsl_matrix *vecRho_n_n = gsl_matrix_calloc((Rho->size1 - 1) * (Rho->size2 - 1), 1);
    gsl_matrix2vector(vecRho_n_n, &rho_n_n.matrix);

    matrix_multiply(ZnonOZnon, vecRho_n_n, Enon, 1, 0, CblasNoTrans, CblasNoTrans);

    gsl_matrix_free(ZnonOZnon);
    gsl_matrix_free(rhocy);
    gsl_matrix_free(vecRho_n_n);
}


/**
 * Generate random number in range 0 to 1
 * @return a random number in range 0 to 1
 */
double rand01() {
    // initialize the random number generator with time-dependent seed
    seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};
    mt19937_64 rng(ss);

    // initialize a uniform distribution between 0 and 1
    uniform_real_distribution<double> uniform(0, 1);

    return uniform(rng);
}

/**
 * Print the iteration number in log file
 * @param iterationNum current iteration number
 */
void print_iteration_num(int iterationNum) {
    LOG(OUTPUT_NORMAL, "Start iteration %d", iterationNum);
    matrixOut << "\n\nStart iteration " << to_string(iterationNum) << endl;
}

/**
 * Print the matrix in log file
 * @param matrix Input, matrix to print
 * @param name Input, name of the matrix
 * @param entryPerRow Input(optional), defines the number of element printed per row in log file
 */
void print_matrix(const gsl_matrix *matrix, const string &name, size_t entryPerRow) {
    if (entryPerRow == 0) {
        // use default value
        entryPerRow = matrix->size2;
    }
    size_t counter = 0;

    matrixOut << endl << name << endl;
    for (int i = 0; i < matrix->size1; i++) {
        for (int j = 0; j < matrix->size2; j++) {
            matrixOut << to_string(gsl_matrix_get(matrix, i, j)) << ", ";
            counter++;
        }

        if (counter == entryPerRow) {
            matrixOut << endl;
            counter = 0;
        }
    }
    matrixOut.flush();
}

/**
 * Print the matrix in log file
 * @param matrix Input, matrix to print
 * @param name Input, name of the matrix
 * @param rowNum Input, number of rows in the matrix
 * @param columnNum Input, number of columns in the matrix
 */
void print_matrix(const gsl_matrix **matrix, const string &name, int rowNum, int columnNum) {
    matrixOut << endl << name << endl;
    for (int i = 0; i < rowNum; i++) {
        const gsl_matrix *row = matrix[i];
        for (int j = 0; j < columnNum; j++) {
            matrixOut << to_string(gsl_matrix_get(row, j, 0)) << ", ";

            if (j == columnNum - 1) {
                matrixOut << endl;
            }
        }
    }
    matrixOut.flush();
}

/**
 * Initialize the log file output stream
 * @param exeName Input, name of the log file
 */
void init_util_functions(const string &exeName, const std::string &detail) {
    timeSeed = chrono::high_resolution_clock::now().time_since_epoch().count();
    string fileName = exeName.substr(0, exeName.find('.'));
    if(!detail.empty()){
        fileName = fileName + "_" + detail;
    }
    matrixOut.open(fileName + "_matrix_log");
}


void matrix_compare(const gsl_matrix *A, const gsl_matrix *B) {
    for (int i = 0; i < A->size1; i++) {
        for (int j = 0; j < A->size2; j++) {
            if (abs(gsl_matrix_get(A, i, j) - gsl_matrix_get(B, i, j)) > 0.0000001) {
                cout << "different res at " << "[" << to_string(i) << "," << to_string(j) << "]  "
                     << to_string(gsl_matrix_get(A, i, j)) << "  "
                     << to_string(gsl_matrix_get(B, i, j)) << endl;
            }
        }
    }
}

void compute_full_eta(const gsl_matrix *Z, const gsl_matrix *Rho, gsl_matrix *eta) {
    gsl_matrix *etaKK = gsl_matrix_calloc(Z->size1, Z->size1);

    gpuBoostedComputeFullEta(Z, Rho, etaKK);

    for (int i = 0; i < etaKK->size1; i++) {
        for (int j = 0; j < etaKK->size2; j++) {
            gsl_matrix_set(eta, i * etaKK->size2 + j, 0, gsl_matrix_get(etaKK, i, j));
        }
    }
    gsl_matrix_free(etaKK);
}

/**
 * Helper function for parallelized kronecker product computation
 * @param prod Output, stores the result of A kron B
 * @param a Input, matrix A
 * @param b Input, matrix B
 * @param startJ Input, start index assigned to this thread
 * @param endJ Input, end index assigned to this thread
 */
void kron_parallel_array(double *prod, const gsl_matrix *a, const gsl_matrix *b, size_t startJ, size_t endJ) {
    unsigned prodWidth = a->size2 * b->size2;
    for (unsigned int i = 0; i < a->size1; ++i) {
        for (unsigned int j = startJ; j < endJ; ++j) {
            double factor = gsl_matrix_get(a, i, j);
            unsigned int prodBaseI = i * b->size1;
            unsigned int prodBaseJ = j * b->size2;
            for (unsigned int s = 0; s < b->size1; s++) {
                for (unsigned int t = 0; t < b->size2; t++) {
                    unsigned int prodI = prodBaseI + s;
                    unsigned int prodJ = prodBaseJ + t;
                    prod[prodI * prodWidth + prodJ] = factor * gsl_matrix_get(b, s, t);
                }
            }
        }
    }
}

/**
 * Compute the Kronecker product of A and B
 * @param prod Output, stores the result of A Kron B
 * @param a Input, matrix A
 * @param b Input, matrix B
 */
void kronecker_product_array(double *prod, const gsl_matrix *a, const gsl_matrix *b) {
    if (a->size2 > 600 || b->size2 > 600) {
        thread t1(kron_parallel_array, prod, a, b, 0, a->size2 / 3);
        thread t2(kron_parallel_array, prod, a, b, a->size2 / 3, a->size2 / 3 * 2);
        thread t3(kron_parallel_array, prod, a, b, a->size2 / 3 * 2, a->size2);
        t1.join();
        t2.join();
        t3.join();
    } else {
        kron_parallel_array(prod, a, b, 0, a->size2);
    }
}


void rank_one_update_eta(int K, int N, int n, gsl_matrix *Z, gsl_matrix *zn, gsl_matrix *Rho, gsl_matrix *Eta,
                         gsl_matrix *Etanon) {
    auto *znkZ = new double[N * K * K];
    auto *Zkzn = new double[N * K * K];
    auto *znkzn = new double[K * K];
    kronecker_product_array(znkZ, zn, Z);
    kronecker_product_array(Zkzn, Z, zn);
    kronecker_product_array(znkzn, zn, zn);

    // whole column
    gsl_matrix_view rho_col = gsl_matrix_submatrix(Rho, 0, n, N, 1);
    // whole row
    gsl_matrix_view rho_row = gsl_matrix_submatrix(Rho, n, 0, 1, N);
    double rho_nn = gsl_matrix_get(Rho, n, n);

    gpuBoostedEtaUpdate(N, K, znkZ, Zkzn, znkzn, &rho_col.matrix, &rho_row.matrix, rho_nn, Eta, Etanon);

    delete[] znkZ;
    delete[] Zkzn;
    delete[] znkzn;
}