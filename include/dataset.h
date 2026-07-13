#pragma once
#include "matrix.h"
#include <vector>
#include <string>

class Dataset {
public:
    Matrix inputs;   // Feature vector (flattened)
    Matrix targets;  // One‑hot or label matrix
};