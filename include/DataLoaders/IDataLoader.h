#pragma once
#include <string>
#include "../dataset.h"

class IDataLoader {
private:
    Dataset dataset;
public:
    virtual ~IDataLoader() = default;
    virtual void load(const std::string& path) = 0;
    virtual Dataset getDataset() const = 0;
    virtual size_t getSize() const = 0;
};
