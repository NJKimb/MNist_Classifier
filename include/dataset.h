#pragma once
#include "matrix.h"

class Dataset {
public:
    Matrix inputs;   // Feature vector (flattened)
    Matrix targets;  // One‑hot or label matrix
};