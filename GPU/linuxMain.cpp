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

int getArraySize(const string &s, const int s_length) {
    char * c = new char[s_length + 1];

    strcpy(c, s.c_str());
    int counter = 0;
    const char *delim = "\t";
    char *p = strtok(c, delim);
    while (p) {
        counter++;
        p = strtok(nullptr, delim);
    }

    delete[] c;
    return counter;
}


int main() {
    char NETin = 'b';
    int N;
    int D;
    // k need to be larger than 1
    int K = 7;
    double F = 1.0;
    int bias = 1;           // 1 = fix first feature to be active for all patients
    double s2u = 0.005;     // auxiliary noise
    double s2B = 0.2;       // noise variance for feature values
    double s2H = 0.001;
    double alpha = 0.5;     // mass parameter for the Indian Buffet Process
    int Nsim = 1000;        // number of algorithm iterations (for Gibbs sampler)
    int maxK = 15;          // maximum number of latent features for memory allocation
    double missing = -1;
    int verbose = 1;

    // ---------------------------------- Load data from txt file -------------------------------------

    ifstream adjIn, attributeIn;
    vector<string> temp;
    string s;
    const char *delim = "\t";

    // todo please enter the correct file location
    adjIn.open(R"(dataSet/Adjacency_matrix.txt)", ios::in);
    while (getline(adjIn, s)) {
        temp.emplace_back(s);
    }
    adjIn.close();

    s = temp.at(0);
    N = getArraySize(s, s.length());
    int **adjacencyMatrix = new int*[N];
    for (int i = 0; i < N; i++) {
        adjacencyMatrix[i] = new int[N];
    }


    char * c = new char[s.length() + 1];

    for (int i = 0; i < N; i++) {
        strcpy(c, temp.at(i).c_str());
        char *p = strtok(c, delim);
        adjacencyMatrix[i][0] = stoi(p);

        for (int j = 1; j < N; j++) {
            p = strtok(nullptr, delim);
            adjacencyMatrix[i][j] = stoi(p);
        }
    }


    temp.clear();
    s = "";
    // todo please enter the correct file location
    attributeIn.open(R"(dataSet/Attribute_matrix.txt)", ios::in);
    while (getline(attributeIn, s)) {
        temp.emplace_back(s);
    }
    attributeIn.close();


    s = temp.at(0);
    D = getArraySize(s, s.length());
    auto **attribute = new double*[N];
    for (int i = 0; i < N; i++) {
        attribute[i] = new double[D];
    }
    char * c2 = new char[s.length() + 100];

    for (int i = 0; i < temp.size(); i++) {
        strcpy(c2, temp.at(i).c_str());
        char *p = strtok(c2, delim);
        attribute[i][0] = strtod(p, nullptr);

        for (int j = 1; j < D; j++) {
            p = strtok(nullptr, delim);
            attribute[i][j] = strtod(p, nullptr);
        }
    }


    // ---------------------------------- Create Fin -------------------------------------
    auto * Fin = new double[D];
    for (int i = 0; i < D; i++) {
        Fin[i] = 1;
    }

    // ----------------------------- Get attribute transpose ------------------------------
    auto * X = new double[D * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            X[j * N + i] = attribute[i][j];
        }
    }

    for(int i = 0; i < N; i++){
        delete[] attribute[i];
    }
    delete[] attribute;

    // -------------------------------- Create Zin -------------------------------------

    // fake random generate
    srand(1);
    int random;
    auto * Z = new double [K * N];
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            random = rand() % 5;
            if (random <= 1) {
                // 40 % chance 0
                Z[i * N + j] = 0.0;
            } else {
                // 60 % chance 1
                Z[i * N + j] = 1.0;
            }
        }
    }

    // ---------------------------------- Create Cin --------------------------------------
    char * Cin = new char[D + 1];
    for (int i = 0; i < D; i++) {
        Cin[i] ='g';
    }
    Cin[D] = '\0';

    auto* A = new double[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = adjacencyMatrix[i][j];
        }
    }
    for (int i = 0; i < N; i++) {
        delete[] adjacencyMatrix[i];
    }
    delete[] adjacencyMatrix;


    infer(X, Cin, Z, NETin, (double *) A, Fin, N, D, K, F,
                             bias, s2u, s2B, s2H, alpha, Nsim,
                             maxK, missing, verbose);

    delete[] X;
    delete[] A;
    delete[] Z;
    delete[] c;
    delete[] Fin;
    delete[] Cin;
    delete[] c2;
    return 0;
}
