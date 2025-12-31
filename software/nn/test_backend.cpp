#include <iostream>

#include "Tensor.hpp"

int main()
{
    try
    {
        // Test basic tensor creation with backend
        nn::Tensor t(2, 3);
        std::cout << "Tensor created with shape: " << t.rows() << "x" << t.cols() << std::endl;

        // Test element access
        t.at(0, 0) = 1.0f;
        t.at(0, 1) = 2.0f;
        std::cout << "Element at (0,0): " << t.at(0, 0) << std::endl;
        std::cout << "Element at (0,1): " << t.at(0, 1) << std::endl;

        // Test operations
        nn::Tensor t2 = t.add(t);
        std::cout << "After adding tensor to itself, element (0,0): " << t2.at(0, 0) << std::endl;

        // Test relu
        nn::Tensor t3 = t.relu();
        std::cout << "After relu, element (0,0): " << t3.at(0, 0) << std::endl;

        std::cout << "Backend interface test completed successfully!" << std::endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}