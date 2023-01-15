#include <stdio.h>
#include "gsl/gsl_statistics_float.h"
#include <fstream>
#include <istream>
#include <string>
#include <cstring>
#include <vector>
#include "ExtendFunction.h"

using namespace std;

/**
 *  This file is the C++ version entry point
 *  You need to edit the file location in order to make it work
 *
 *
 * */

int getArraySize(const string &s) {
    char c[s.length() + 1];
    strcpy(c, s.c_str());
    int counter = 0;
    const char *delim = "\t";
    char *p = strtok(c, delim);
    while (p) {
        counter++;
        p = strtok(nullptr, delim);
    }
    return counter;
}


int main() {
    char NETin = 'b';
    int N;
    int D;
    // k need to be larger than 1
    int K = 2;
    double F = 1.0;
    int bias = 1;           // 1 = fix first feature to be active for all patients
    double s2u = 0.005;     // auxiliary noise
    double s2B = 0.2;       // noise variance for feature values
    double s2H = 0.001;
    double alpha = 0.5;     // mass parameter for the Indian Buffet Process
    int Nsim = 1000;        // number of algorithm iterations (for Gibbs sampler)
    int maxK = 25;          // maximum number of latent features for memory allocation
    double missing = -1;
    int verbose = 1;

    // ---------------------------------- Load data from txt file -------------------------------------

    ifstream adjIn, attributeIn;
    vector<string> temp;
    string s;
    const char *delim = "\t";

    // todo please enter the correct file location
    adjIn.open(R"(Adjacency_matrix.txt)", ios::in);
    while (getline(adjIn, s)) {
        temp.emplace_back(s);
    }
    adjIn.close();

    s = temp.at(0);
    int adjCol = getArraySize(s);
    int adjRow = (int) temp.size();
    int **adjacencyMatrix = (int **) malloc(sizeof(int *) * adjRow);
    for (int i = 0; i < adjRow; i++) {
        adjacencyMatrix[i] = (int *) malloc(sizeof(int) * adjCol);
    }


    char c[s.length() + 1];

    for (int i = 0; i < adjRow; i++) {
        strcpy(c, temp.at(i).c_str());
        char *p = strtok(c, delim);
        adjacencyMatrix[i][0] = stoi(p);

        for (int j = 1; j < adjCol; j++) {
            p = strtok(nullptr, delim);
            adjacencyMatrix[i][j] = stoi(p);
        }
    }


    temp.clear();
    s = "";
    // todo please enter the correct file location
    attributeIn.open(R"(Attribute_matrix.txt)", ios::in);
    while (getline(attributeIn, s)) {
        temp.emplace_back(s);
    }
    attributeIn.close();


    s = temp.at(0);
    int size = getArraySize(s);
    double attribute[temp.size()][size];
    char c2[s.length() + 100];

    for (int i = 0; i < temp.size(); i++) {
        strcpy(c2, temp.at(i).c_str());
        char *p = strtok(c2, delim);
        attribute[i][0] = strtod(p, nullptr);

        for (int j = 1; j < size; j++) {
            p = strtok(nullptr, delim);
            attribute[i][j] = strtod(p, nullptr);
        }
    }

    // ---------------------------------- Get N and D -------------------------------------

    N = (int) temp.size();
    D = size;

    // ---------------------------------- Create Fin -------------------------------------
    double Fin[D];
    for (int i = 0; i < D; i++) {
        Fin[i] = 1;
    }

    // ----------------------------- Get attribute transpose ------------------------------
    double Xin[D][N];
    double X[D * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            Xin[j][i] = attribute[i][j];
            X[j * N + i] = Xin[j][i];
        }
    }

    // -------------------------------- Create Zin -------------------------------------

    // fake random generate
    srand(1);
    int random;
    double Zin[K][N];
    double Z[K * N];
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            random = rand() % 5;
            if (random <= 1) {
                // 40 % chance 0
                Zin[i][j] = 0.0;
            } else {
                // 60 % chance 1
                Zin[i][j] = 1.0;
            }
            Z[i * N + j] = Zin[i][j];
        }
    }

    // ---------------------------------- Create Cin --------------------------------------
    char Cin[D];
    for (int i = 0; i < D; i++) {
        Cin[i] = 'g';
    }

    auto* A = (double*)malloc(sizeof (double ) * adjRow * adjCol);
    for (int i = 0; i < adjRow; i++) {
        for (int j = 0; j < adjCol; j++) {
            A[i * adjCol + j] = adjacencyMatrix[i][j];
        }
    }
    for (int i = 0; i < adjRow; i++) {
        free(adjacencyMatrix[i] );
    }
    free(adjacencyMatrix);


    InferReturn *res = infer(X, Cin, Z, NETin, (double *) A, Fin, N, D, K, F,
                             bias, s2u, s2B, s2H, alpha, Nsim,
                             maxK, missing, verbose);

    free(A);
    printf("finished with %f", res->s2Rho);
    return 0;
}
