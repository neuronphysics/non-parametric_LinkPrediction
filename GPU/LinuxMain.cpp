#include "ExtendFunction.h"

using namespace std;

/**
 *  This file is the C++ version entry point
 *  You need to edit the file location in order to make it work
 *
 *
 * */

int getArraySize(const string &s, const int s_length) {
    char *c = new char[s_length + 1];

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
    LOG(OUTPUT_NORMAL, "Version description:")
    LOG(OUTPUT_NORMAL, VERSION_DECLARE)

    char NETin = 'b';
    int N;
    int D;
    // k need to be larger than 1
    int K = 3;
    double F = 1.0;
    int bias = 1;           // 1 = fix first feature to be active for all patients
    double s2u = 0.01;     // auxiliary noise
    double s2B = 0.2;       // noise variance for feature values
    double s2H = 0.1;
    double alpha = 10;     // mass parameter for the Indian Buffet Process
    int Nsim = 1000;        // number of algorithm iterations (for Gibbs sampler)
    int maxK = 20;          // maximum number of latent features for memory allocation
    double missing = -1;

    // ---------------------------------- Load data from txt file -------------------------------------
    string dataSetName = DATASET_NAME;
    init_util_functions(dataSetName, "test_1");
    string filePath = R"(dataSet/)";
    string adjFileName = "Adjacency_matrix";
    string attrFileName = "Attribute_matrix";


    vector<string> lines;
    string line;
    const char *delim = "\t";


    ifstream adjIn(filePath + adjFileName + dataSetName);
    ifstream attributeIn(filePath + attrFileName + dataSetName);

    // check if the file open properly
    if (!adjIn.is_open() || !attributeIn.is_open()) {
        LOG(OUTPUT_NORMAL, "Open file fail")
        return 1;
    }

    while (getline(adjIn, line)) {
        lines.emplace_back(line);
    }
    adjIn.close();

    line = lines.at(0);
    N = getArraySize(line, line.length());
    int **adjacencyMatrix = new int *[N];
    for (int i = 0; i < N; i++) {
        adjacencyMatrix[i] = new int[N];
    }


    char *c = new char[line.length() + 1];

    for (int i = 0; i < N; i++) {
        strcpy(c, lines.at(i).c_str());
        char *p = strtok(c, delim);
        adjacencyMatrix[i][0] = stoi(p);

        for (int j = 1; j < N; j++) {
            p = strtok(nullptr, delim);
            adjacencyMatrix[i][j] = stoi(p);
        }
    }


    lines.clear();
    line = "";
    while (getline(attributeIn, line)) {
        lines.emplace_back(line);
    }
    attributeIn.close();


    line = lines.at(0);
    D = getArraySize(line, line.length());
    auto **attribute = new double *[N];
    for (int i = 0; i < N; i++) {
        attribute[i] = new double[D];
    }
    char *c2 = new char[line.length() + 100];

    for (int i = 0; i < lines.size(); i++) {
        strcpy(c2, lines.at(i).c_str());
        char *p = strtok(c2, delim);
        attribute[i][0] = strtod(p, nullptr);

        for (int j = 1; j < D; j++) {
            p = strtok(nullptr, delim);
            attribute[i][j] = strtod(p, nullptr);
        }
    }


    // ---------------------------------- Create Fin -------------------------------------
    auto *Fin = new double[D];
    for (int i = 0; i < D; i++) {
        Fin[i] = 1;
    }

    // ----------------------------- Get attribute transpose ------------------------------
    auto *X = new double[D * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            X[j * N + i] = attribute[i][j];
        }
    }

    for (int i = 0; i < N; i++) {
        delete[] attribute[i];
    }
    delete[] attribute;

    // -------------------------------- Create Zin -------------------------------------

    // fake random generate
    int random;
    auto *Z = new double[K * N];
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
    char *Cin = new char[D + 1];
    for (int i = 0; i < D; i++) {
        Cin[i] = 'g';
    }
    Cin[D] = '\0';

    auto *A = new double[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = adjacencyMatrix[i][j];
        }
    }
    for (int i = 0; i < N; i++) {
        delete[] adjacencyMatrix[i];
    }
    delete[] adjacencyMatrix;


    delete[] c2;
    delete[] c;

    infer(X, Cin, Z, NETin, (double *) A, Fin, N, D, K, F,
          bias, s2u, s2B, s2H, alpha, Nsim,
          maxK, missing);

    delete[] X;
    delete[] A;
    delete[] Z;
    delete[] Fin;
    delete[] Cin;

    return 0;
}
