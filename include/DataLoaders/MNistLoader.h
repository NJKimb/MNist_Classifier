#pragma once
#include "dataset.h"

class MNISTLoader : public IDataLoader {
private:
    size_t size;
public:
    MNISTLoader();

    void load(const std::string& path) override;

    DataSample get(size_t idx) const override;

    size_t getSize() const override {return size;};
};
