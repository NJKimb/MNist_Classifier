#pragma once
#include <functional>
#include <vector>

class Matrix {
private:
    int rows;
    int columns;
    std::vector<double> data;
public:
    Matrix(int r, int c);
    Matrix();
    Matrix(int r, int c, std::vector<double> initData);

    void Randomize();

    double& operator()(int r, int c);
    double operator()(int r, int c) const;

    void add(double n);
    void add(const Matrix& m);

    static Matrix subtract(const Matrix& A, const Matrix& b);

    void multiply(double n);
    void multiply(const Matrix& m);
    static Matrix dot(const Matrix& a, const Matrix& b);
    static Matrix hadamard(const Matrix& a, const Matrix& b);

    Matrix transpose() const;

    void map(std::function<double(double)> func);
    static Matrix map (const Matrix& m, std::function<double(double)> func);

    int getRows() const {return rows;};
    int getColumns() const {return columns;};

    void printMatrix();
};
