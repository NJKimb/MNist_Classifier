#include <iostream>
#include <string>
#include "DataLoaders/MNistLoader.h"

int main() {
    MNISTLoader mnist; 
    mnist.load("mnist_dataset/");
    
    Dataset data = mnist.getDataset(); 
}
