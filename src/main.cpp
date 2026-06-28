#include <iostream>

#include "matrix.h"

int main() {
    Matrix matrix1(3,3);
    matrix1.Randomize();
    matrix1.printMatrix();

    Matrix matrix2(3,3);
    matrix2.Randomize();
    matrix2.printMatrix();

    Matrix result(3,3);
    result = Matrix::dot(matrix1, matrix2);

    result.printMatrix();
    return 0;
}
