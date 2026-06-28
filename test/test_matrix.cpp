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