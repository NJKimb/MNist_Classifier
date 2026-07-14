#pragma once
#include <functional>
#include <vector>
#include <random>

class Matrix {
private:
    size_t rows;
    size_t columns;
    std::vector<double> data;

    inline static thread_local std::mt19937 generator = std::mt19937(std::random_device{}());
public:
    Matrix();
    Matrix(size_t r, size_t c);
    Matrix(size_t r, size_t c, std::vector<double> initData);

    void Randomize();
    void SetZero();
    void resize(size_t r, size_t c);

    double& operator()(size_t r, size_t c);
    double operator()(size_t r, size_t c) const;

    void add(double n);
    void add(const Matrix& m);

    static Matrix subtract(const Matrix& A, const Matrix& b);

    static Matrix dot(const Matrix& a, const Matrix& b);
    static Matrix hadamard(const Matrix& a, const Matrix& b);

    Matrix transpose() const;

    static Matrix map (const Matrix& m, const std::function<double(double)>& func);

    size_t getRows() const {return rows;};
    size_t getColumns() const {return columns;};
    size_t getSize() const {return data.size();};

    void printMatrix();
};
