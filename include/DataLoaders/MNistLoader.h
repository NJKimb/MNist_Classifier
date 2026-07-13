#pragma once
#include "IDataLoader.h"
#include <bit>
#include <cstdint>

// Reverse bytes for MNIST data, which is stored in big-endian format
inline uint32_t reverseBytes(uint32_t val) {
    if constexpr (std::endian::native == std::endian::little) {
        return std::byteswap(val);
    }
    return val;
}

class MNISTLoader : public IDataLoader {
private:
    Dataset dataset;
    size_t size;
public:
    MNISTLoader();

    void load(const std::string& path) override;

    Dataset getDataset() const override {return dataset;};

    size_t getSize() const override {return size;};
};
