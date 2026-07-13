#pragma once
#include "matrix.h"
#include <vector>
#include <string>

class DataSample {
public:
    Matrix input;   // Feature vector (flattened)
    Matrix target;  // One‑hot or label matrix
};

class IDataLoader {
public:
    virtual ~IDataLoader() = default;

    // Load data from file(s); may throw on I/O errors.
    virtual void load(const std::string& path) = 0;

    // Return the total number of samples available
    virtual size_t getSize() const = 0;

    // Provide access to a single sample by index
    virtual DataSample get(size_t idx) const = 0;
};