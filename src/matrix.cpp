#include "matrix.h"
#include <iostream>
#include <random>

// Default constructor
Matrix::Matrix() {
  rows = 0;
  columns = 0;
}

// Constructor for matrix of size r x c
Matrix::Matrix(size_t r, size_t c) {
  rows = r;
  columns = c;
  data.resize(r * c);
}

// Constructor for matrix of size r x c with initial data from vector
Matrix::Matrix(size_t r, size_t c, std::vector<double> initData) {
  rows = r;
  columns = c;
  data = std::move(initData);
}

// Add scalar to each element of the matrix
void Matrix::add(double n) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      (*this)(i, j) += n;
    }
  }
}

// Add two matricies together element-wise
void Matrix::add(const Matrix &m) {
  if (rows != m.rows || columns != m.columns) {
    throw std::invalid_argument("Incompatible dimensions for matrix addition.");
  }
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      (*this)(i, j) += m(i, j);
    }
  }
}

// Subtract matrix b from matrix a element-wise
Matrix Matrix::subtract(const Matrix &a, const Matrix &b) {
  if (a.rows != b.rows || a.columns != b.columns) {
    throw std::invalid_argument(
        "Incompatible dimensions for matrix subtraction.");
  }
  Matrix result(a.rows, a.columns);
  for (size_t i = 0; i < a.rows; i++) {
    for (size_t j = 0; j < a.columns; j++) {
      result(i, j) = a(i, j) - b(i, j);
    }
  }
  return result;
}

// Transpose a matrix
Matrix Matrix::transpose() const {
  Matrix result(columns, rows);
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < columns; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

// Apply a n activation function to each element of a matrix
Matrix Matrix::map(const Matrix &m, const std::function<double(double)>& func) {
  Matrix result(m.rows, m.columns);
  for (size_t i = 0; i < m.rows; i++) {
    for (size_t j = 0; j < m.columns; j++) {
      result(i, j) = func(m(i, j));
    }
  }
  return result;
}

// Randomize all values in the matrix with values between 0 and 1
void Matrix::Randomize() {
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (double &i : data) {
    i = dis(generator);
  }
}

// Set all values in the matrix to 0
void Matrix::SetZero() { std::fill(data.begin(), data.end(), 0.0); }

// Resize matrix to given dimensions
void Matrix::resize(size_t r, size_t c) {
  rows = r;
  columns = c;
  data.resize(r * c);
}

// Hadamard product (element-wise multiplication)
Matrix Matrix::hadamard(const Matrix &a, const Matrix &b) {
  if (a.getRows() != b.getRows() || a.getColumns() != b.getColumns()) {
    throw std::invalid_argument(
        "Incompatible dimensions for Hadamard product.");
  }

  Matrix result(a.getRows(), a.getColumns());
  for (size_t i = 0; i < a.getRows(); ++i) {
    for (size_t j = 0; j < a.getColumns(); ++j) {
      result(i, j) = a(i, j) * b(i, j);
    }
  }
  return result;
}

/* Overload operator to access element [r,c] since the matrix
   is stored as a 1D array. For example, to access [2,2] in a 3x3
   matrix 1 * 3 + 1 would access the 4th element in an array of size
   9, with 1 subtracted from r and c to account for 0 indexing.
*/
double &Matrix::operator()(size_t r, size_t c) {
  return data[r * columns + c];
}

double Matrix::operator()(size_t r, size_t c) const {
  return data[r * columns + c];
}

// Dot product of two matrices (static)
Matrix Matrix::dot(const Matrix &a, const Matrix &b) {
  if (a.getColumns() != b.getRows()) {
    throw std::invalid_argument(
        "Incompatible dimensions for matrix dot product.");
  }

  Matrix result(a.getRows(), b.getColumns());
  for (size_t i = 0; i < a.getRows(); ++i) {
    for (size_t j = 0; j < b.getColumns(); ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < a.getColumns(); ++k) {
        sum += a(i, k) * b(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}

// Print matrix
void Matrix::printMatrix() {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      std::cout << (*this)(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
