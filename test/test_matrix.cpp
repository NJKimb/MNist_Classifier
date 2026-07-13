// test_matrix.cpp
#define CATCH_CONFIG_MAIN   // Generates the main() automatically
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>
#include "../include/matrix.h"

TEST_CASE("Matrix construction and size") {
    Matrix m(3,4);
    REQUIRE(m.getRows() == 3);
    REQUIRE(m.getColumns() == 4);

    // All elements should be zero initially
    for (int i=0; i<12; ++i) {
        REQUIRE(m(i/4,i%4) == Catch::Approx(0.0));
    }
}

TEST_CASE("Randomization changes values") {
    Matrix m(2,2);
    m.Randomize();
    double v1 = m(0,0);
    double v2 = m(0,1);

    // They should be in [0,1] and usually different
    REQUIRE(v1 >= 0.0);
    REQUIRE(v1 <= 1.0);
    REQUIRE(v2 >= 0.0);
    REQUIRE(v2 <= 1.0);
    if (m.getRows() * m.getColumns() > 1) {
        // It's extremely unlikely to get exactly the same number twice
        REQUIRE(v1 != Catch::Approx(v2));
    }
}

TEST_CASE("Dot product correctness") {
    Matrix a(2,3); a.Randomize();
    Matrix b(3,4); b.Randomize();

    Matrix c = Matrix::dot(a,b);
    REQUIRE(c.getRows() == 2);
    REQUIRE(c.getColumns() == 4);

    // Verify one entry by hand
    double manual_sum = 0.0;
    for (int k=0; k<3; ++k) {
        manual_sum += a(0,k) * b(k,1);
    }
    REQUIRE(c(0,1) == Catch::Approx(manual_sum));
}

TEST_CASE("Matrix constructor with initial data") {
    Matrix m(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    REQUIRE(m.getRows() == 2);
    REQUIRE(m.getColumns() == 3);
    
    REQUIRE(m(0, 0) == Catch::Approx(1.0));
    REQUIRE(m(0, 1) == Catch::Approx(2.0));
    REQUIRE(m(0, 2) == Catch::Approx(3.0));
    REQUIRE(m(1, 0) == Catch::Approx(4.0));
    REQUIRE(m(1, 1) == Catch::Approx(5.0));
    REQUIRE(m(1, 2) == Catch::Approx(6.0));
}

TEST_CASE("Matrix scalar and matrix addition") {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});
    
    // Add scalar
    m.add(2.0);
    REQUIRE(m(0, 0) == Catch::Approx(3.0));
    REQUIRE(m(0, 1) == Catch::Approx(4.0));
    REQUIRE(m(1, 0) == Catch::Approx(5.0));
    REQUIRE(m(1, 1) == Catch::Approx(6.0));

    // Add matrix
    m.add(Matrix(2, 2, {1.0, -1.0, 2.0, -2.0}));
    REQUIRE(m(0, 0) == Catch::Approx(4.0));
    REQUIRE(m(0, 1) == Catch::Approx(3.0));
    REQUIRE(m(1, 0) == Catch::Approx(7.0));
    REQUIRE(m(1, 1) == Catch::Approx(4.0));

    // Dimensions check
    REQUIRE_THROWS_AS(m.add(Matrix(2, 3)), std::invalid_argument);
}

TEST_CASE("Matrix subtraction") {
    Matrix a(2, 2, {5.0, 8.0, 2.0, 1.0});
    Matrix b(2, 2, {1.0, 3.0, 4.0, -2.0});

    Matrix c = Matrix::subtract(a, b);
    REQUIRE(c(0, 0) == Catch::Approx(4.0));
    REQUIRE(c(0, 1) == Catch::Approx(5.0));
    REQUIRE(c(1, 0) == Catch::Approx(-2.0));
    REQUIRE(c(1, 1) == Catch::Approx(3.0));

    Matrix invalid(2, 3);
    REQUIRE_THROWS_AS(Matrix::subtract(a, invalid), std::invalid_argument);
}

TEST_CASE("Matrix scalar and element-wise multiplication") {
    Matrix m(2, 2, {1.0, 2.0, 3.0, 4.0});

    // In-place scalar multiplication
    m.multiply(3.0);
    REQUIRE(m(0, 0) == Catch::Approx(3.0));
    REQUIRE(m(0, 1) == Catch::Approx(6.0));
    REQUIRE(m(1, 0) == Catch::Approx(9.0));
    REQUIRE(m(1, 1) == Catch::Approx(12.0));

    // In-place matrix element-wise multiplication
    m.multiply(Matrix(2, 2, {2.0, 0.5, -1.0, 2.0}));
    REQUIRE(m(0, 0) == Catch::Approx(6.0));
    REQUIRE(m(0, 1) == Catch::Approx(3.0));
    REQUIRE(m(1, 0) == Catch::Approx(-9.0));
    REQUIRE(m(1, 1) == Catch::Approx(24.0));

    // Dimensions check
    REQUIRE_THROWS_AS(m.multiply(Matrix(3, 2)), std::invalid_argument);
}

TEST_CASE("Static element-wise matrix multiplication (Hadamard)") {
    Matrix a(2, 2, {1.0, 2.0, 3.0, 4.0});
    Matrix b(2, 2, {2.0, 0.5, 3.0, -1.0});

    Matrix c = Matrix::hadamard(a, b);
    REQUIRE(c(0, 0) == Catch::Approx(2.0));
    REQUIRE(c(0, 1) == Catch::Approx(1.0));
    REQUIRE(c(1, 0) == Catch::Approx(9.0));
    REQUIRE(c(1, 1) == Catch::Approx(-4.0));

    Matrix invalid(2, 3);
    REQUIRE_THROWS_AS(Matrix::hadamard(a, invalid), std::invalid_argument);
}

TEST_CASE("Static matrix dot product") {
    Matrix a(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix b(3, 2, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    Matrix c = Matrix::dot(a, b);
    REQUIRE(c.getRows() == 2);
    REQUIRE(c.getColumns() == 2);

    // 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    REQUIRE(c(0, 0) == Catch::Approx(58.0));
    // 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    REQUIRE(c(0, 1) == Catch::Approx(64.0));
    // 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    REQUIRE(c(1, 0) == Catch::Approx(139.0));
    // 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    REQUIRE(c(1, 1) == Catch::Approx(154.0));

    Matrix invalid(2, 2);
    REQUIRE_THROWS_AS(Matrix::dot(a, invalid), std::invalid_argument);
}

TEST_CASE("Matrix transpose") {
    Matrix m(2, 3, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Matrix t = m.transpose();

    REQUIRE(t.getRows() == 3);
    REQUIRE(t.getColumns() == 2);
    
    REQUIRE(t(0, 0) == Catch::Approx(1.0));
    REQUIRE(t(0, 1) == Catch::Approx(4.0));
    REQUIRE(t(1, 0) == Catch::Approx(2.0));
    REQUIRE(t(1, 1) == Catch::Approx(5.0));
    REQUIRE(t(2, 0) == Catch::Approx(3.0));
    REQUIRE(t(2, 1) == Catch::Approx(6.0));
}

TEST_CASE("Matrix in-place mapping function") {
    Matrix m(2, 2, {1.0, -2.0, 3.0, -4.0});
    
    // Absolute value mapping
    m.map([](double x) { return std::abs(x); });
    
    REQUIRE(m(0, 0) == Catch::Approx(1.0));
    REQUIRE(m(0, 1) == Catch::Approx(2.0));
    REQUIRE(m(1, 0) == Catch::Approx(3.0));
    REQUIRE(m(1, 1) == Catch::Approx(4.0));
}
