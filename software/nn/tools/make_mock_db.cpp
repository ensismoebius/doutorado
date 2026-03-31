#include <iostream>
#include <string>

#include "nn/testing/SqliteTestHelpers.hpp"

int main(int argc, char** argv)
{
    try
    {
        int subj = 0;
        std::string path = nn::testing::create_mock_imagined_db(subj, 128, 1024);
        std::cout << path << std::endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
