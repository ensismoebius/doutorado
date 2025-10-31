#include <matioCpp/File.h>

#include <iostream>

using matioCpp::File;

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <mat-file>\n";
        return 1;
    }

    try
    {
        File file(argv[1]);
        auto variables = file.variableNames();

        std::cout << "Variables in file: " << argv[1] << '\n';
        for (const auto& name : variables)
        {
            auto var = file.read(name);
            std::cout << "  " << name << ": " << var.class_name() << " [";
            const auto& dims = var.dimensions();
            for (size_t i = 0; i < dims.size(); ++i)
            {
                if (i != 0)
                {
                    std::cout << "x";
                }
                std::cout << dims[i];
            }
            std::cout << "]\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
