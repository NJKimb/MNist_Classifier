#include "matrix.h"

#include <iostream>
#include <random>

Matrix::Matrix(const int r, const int c) {
    rows = r;
    columns = c;
    data.resize(r * c);
}

void Matrix::Randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (double & i : data) {
        i = dis(gen);
    }
}

/* Overload operator to access element [r,c] since the matrix
   is stored as a 1D array. For example, to access [2,2] in a 3x3
   matrix 1 * 3 + 1 would access the 4th element in an array of size
   9, with 1 subtracted from r and c to account for 0 indexing.
*/
double& Matrix::operator()(const int r, const int c) {
    return data[r * columns + c];
}

double Matrix::operator()(const int r, const int c) const {
    return data[r * columns + c];
}

Matrix Matrix::dot(const Matrix &a, const Matrix &b) {
    if (a.getColumns() != b.getRows()) {
        throw std::invalid_argument("Incompatible dimensions for matrix dot product.");
    }

    Matrix result(a.getRows(), b.getColumns());
    for (int i = 0; i < a.getRows(); ++i) {
        for (int j = 0; j < b.getColumns(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < a.getColumns(); ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

void Matrix::printMatrix() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            std::cout << (*this)(i,j) << " ";
        }
        std::cout << std::endl;

    }
    std::cout << std::endl;
}


