#include "DataLoaders/MNistLoader.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

MNISTLoader::MNISTLoader() {
  this->size = 0;
  this->dataset = Dataset();
}

void MNISTLoader::load(const std::string &directoryPath) {
  std::string imagePath = directoryPath + "train-images.idx3-ubyte";
  std::string labelPath = directoryPath + "train-labels.idx1-ubyte";

  std::ifstream imageFile(imagePath, std::ios::binary);
  std::ifstream labelFile(labelPath, std::ios::binary);

  if (!imageFile.is_open() || !labelFile.is_open()) {
    throw std::runtime_error("Unable to open MNIST files!");
  }

  uint32_t imageMagic, numImages, numRows, numCols;
  uint32_t labelMagic, numLabels;

  // Read header data and convert endianess
  imageFile.read(reinterpret_cast<char *>(&imageMagic), sizeof(imageMagic));
  imageFile.read(reinterpret_cast<char *>(&numImages), sizeof(numImages));
  imageFile.read(reinterpret_cast<char *>(&numRows), sizeof(numRows));
  imageFile.read(reinterpret_cast<char *>(&numCols), sizeof(numCols));

  labelFile.read(reinterpret_cast<char *>(&labelMagic), sizeof(labelMagic));
  labelFile.read(reinterpret_cast<char *>(&numLabels), sizeof(numLabels));

  imageMagic = reverseBytes(imageMagic);
  numImages = reverseBytes(numImages);
  numRows = reverseBytes(numRows);
  numCols = reverseBytes(numCols);

  labelMagic = reverseBytes(labelMagic);
  numLabels = reverseBytes(numLabels);

  // Check magic numbers
  if (imageMagic != 2051) {
    throw std::runtime_error("Invalid magic number for images!");
  }

  if (labelMagic != 2049) {
    throw std::runtime_error("Invalid magic number for labels!");
  }

  if (numImages != numLabels) {
    throw std::runtime_error("Number of images must equal number of labels!");
  }

  // Allocate memory in vector to reduce reallocations
  this->size = numImages;
  this->dataset.inputs.resize(numImages, numRows * numCols);
  this->dataset.targets.resize(numImages, 10);

// Buffers to read raw byte data into (remains efficient for binary disk I/O)
  std::vector<uint8_t> imgBuffer(numImages * numRows * numCols);
  std::vector<uint8_t> labelBuffer(numImages);
  // Read data in one sweep
  if (!imageFile.read(reinterpret_cast<char*>(imgBuffer.data()), imgBuffer.size())) {
    throw std::runtime_error("Failed to read image data");
  }
  if (!labelFile.read(reinterpret_cast<char*>(labelBuffer.data()), labelBuffer.size())) {
    throw std::runtime_error("Failed to read label data");
  }


  for (size_t i = 0; i < numImages; i++) {
    for (size_t j = 0; j < numRows * numCols; j++) {
      this->dataset.inputs(i, j) = 
          imgBuffer[i * numRows * numCols + j] / 255.0;
    }
  }

  // Create one‑hot encoded labels
  this->dataset.targets.SetZero();
  for (size_t i = 0; i < numImages; i++) {
    this->dataset.targets(i, labelBuffer[i]) = 1;
  }

  // Close files
  imageFile.close();
  labelFile.close();

  std::cout << "MNIST dataset loaded successfully: " << numImages
            << " samples\n";
}
