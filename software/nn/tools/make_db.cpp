/**
 * @file tools/make_db.cpp
 * @brief Implementation for Make db.
 *

 */

#include <iostream>
#include <string>

#include "nn/testing/SqliteTestHelpers.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: make_db <db_root>\n";
        return 2;
    }
    try
    {
        nn::testing::create_simple_protocol_db(std::string(argv[1]), 16, 128);
        std::cout << "created db at " << argv[1] << "/database.sqlite\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
