#define CATCH_CONFIG_MAIN // Generates the main() automatically
#include "DataLoaders/MNistLoader.h"
#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>
#include <fstream>
#include <vector>


// Test case for reverseBytes function (swap endianness for MNist data)
TEST_CASE("Endian swapping - 32-bit integers", "[endian]") {
  SECTION("Edge cases: zero and max values") {
    REQUIRE(reverseBytes(0x00000000) == 0x00000000);
    REQUIRE(reverseBytes(0xFFFFFFFF) == 0xFFFFFFFF);
  }

  SECTION("Distinct byte permutation mapping") {
    // Verifies byte 0 -> byte 3, byte 1 -> byte 2, etc.
    REQUIRE(reverseBytes(0x12345678) == 0x78563412);
  }

  SECTION("Single active bytes in different positions") {
    REQUIRE(reverseBytes(0x000000FF) == 0xFF000000);
    REQUIRE(reverseBytes(0x0000FF00) == 0x00FF0000);
    REQUIRE(reverseBytes(0x00FF0000) == 0x0000FF00);
    REQUIRE(reverseBytes(0xFF000000) == 0x000000FF);
  }
  SECTION("Double swapping returns original value") {
    uint32_t original = 0xDEADBEEF;
    REQUIRE(reverseBytes(reverseBytes(original)) == original);
  }
}

// Helper function to dynamically locate the dataset directory
std::string findDatasetPath() {
  std::vector<std::string> candidates = {"mnist_dataset/", "../mnist_dataset/",
                                         "../../mnist_dataset/",
                                         "../../../mnist_dataset/"};
  for (const auto &path : candidates) {
    std::ifstream imgFile(path + "train-images.idx3-ubyte", std::ios::binary);
    std::ifstream lblFile(path + "train-labels.idx1-ubyte", std::ios::binary);
    if (imgFile.good() && lblFile.good()) {
      return path;
    }
  }
  return "";
}

TEST_CASE("MNISTLoader - Load and Validate Dataset", "[loader]") {
  std::string datasetPath = findDatasetPath();
  INFO("Resolved MNIST dataset path: " << datasetPath);
  REQUIRE_FALSE(datasetPath.empty());

  static MNISTLoader loader;
  static bool loaded = false;
  if (!loaded) {
    REQUIRE_NOTHROW(loader.load(datasetPath));
    loaded = true;
  }

  SECTION("Successful load and metadata verification") {
    REQUIRE(loader.getSize() == 60000);

    Dataset dataset = loader.getDataset();
    // Check input matrix shape
    REQUIRE(dataset.inputs.getRows() == 60000);
    REQUIRE(dataset.inputs.getColumns() == 784);

    // Check targets matrix shape
    REQUIRE(dataset.targets.getRows() == 60000);
    REQUIRE(dataset.targets.getColumns() == 10);
  }

  SECTION("Validate known training labels") {
    Dataset dataset = loader.getDataset();

    // The first 10 labels of the MNIST training set are: 5, 0, 4, 1, 9, 2, 1,
    // 3, 1, 4
    std::vector<int> expectedLabels = {5, 0, 4, 1, 9, 2, 1, 3, 1, 4};
    for (size_t i = 0; i < expectedLabels.size(); ++i) {
      int expectedLabel = expectedLabels[i];
      for (int j = 0; j < 10; ++j) {
        if (j == expectedLabel) {
          REQUIRE(dataset.targets(static_cast<int>(i), j) ==
                  Catch::Approx(1.0));
        } else {
          REQUIRE(dataset.targets(static_cast<int>(i), j) ==
                  Catch::Approx(0.0));
        }
      }
    }
  }

  SECTION("Compare loader contents with raw binary files directly") {
    Dataset dataset = loader.getDataset();

    std::ifstream imgFile(datasetPath + "train-images.idx3-ubyte",
                          std::ios::binary);
    std::ifstream lblFile(datasetPath + "train-labels.idx1-ubyte",
                          std::ios::binary);

    REQUIRE(imgFile.is_open());
    REQUIRE(lblFile.is_open());

    // Skip headers
    // Image header: magic (4B), numImages (4B), numRows (4B), numCols (4B) = 16
    // bytes
    imgFile.seekg(16);
    // Label header: magic (4B), numLabels (4B) = 8 bytes
    lblFile.seekg(8);

    // Verify the first 5 samples against raw binary stream data
    for (int i = 0; i < 5; ++i) {
      uint8_t rawLabel = 0;
      lblFile.read(reinterpret_cast<char *>(&rawLabel), 1);

      // Assert targets match label (one-hot encoded)
      for (int j = 0; j < 10; ++j) {
        if (j == rawLabel) {
          REQUIRE(dataset.targets(i, j) == Catch::Approx(1.0));
        } else {
          REQUIRE(dataset.targets(i, j) == Catch::Approx(0.0));
        }
      }

      // Assert inputs match pixel data (normalized)
      std::vector<uint8_t> rawPixels(784);
      imgFile.read(reinterpret_cast<char *>(rawPixels.data()), 784);
      for (int j = 0; j < 784; ++j) {
        double expectedPixel = rawPixels[j] / 255.0;
        REQUIRE(dataset.inputs(i, j) == Catch::Approx(expectedPixel));
      }
    }
  }

  SECTION("Error handling for invalid paths") {
    MNISTLoader badLoader;
    REQUIRE_THROWS_AS(badLoader.load("non_existent_directory/"),
                      std::runtime_error);
  }
}
