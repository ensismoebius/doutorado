#include <matioCpp/File.h>
#include <matioCpp/ForwardDeclarations.h> // For VariableType and ValueType enums

#include <iostream>
#include <string> // For std::string

// Helper function to convert VariableType to string
std::string variableTypeToString(matioCpp::VariableType type)
{
    switch (type)
    {
        case matioCpp::VariableType::CellArray:
            return "CellArray";
        case matioCpp::VariableType::Struct:
            return "Struct";
        case matioCpp::VariableType::StructArray:
            return "StructArray";
        case matioCpp::VariableType::MultiDimensionalArray:
            return "MultiDimensionalArray";
        case matioCpp::VariableType::Vector:
            return "Vector";
        case matioCpp::VariableType::Element:
            return "Element";
        case matioCpp::VariableType::Unsupported:
            return "Unsupported";
    }
    return "UnknownVariableType";
}

// Helper function to convert ValueType to string
std::string valueTypeToString(matioCpp::ValueType type)
{
    switch (type)
    {
        case matioCpp::ValueType::INT8:
            return "INT8";
        case matioCpp::ValueType::UINT8:
            return "UINT8";
        case matioCpp::ValueType::INT16:
            return "INT16";
        case matioCpp::ValueType::UINT16:
            return "UINT16";
        case matioCpp::ValueType::INT32:
            return "INT32";
        case matioCpp::ValueType::UINT32:
            return "UINT32";
        case matioCpp::ValueType::INT64:
            return "INT64";
        case matioCpp::ValueType::UINT64:
            return "UINT64";
        case matioCpp::ValueType::SINGLE:
            return "SINGLE";
        case matioCpp::ValueType::DOUBLE:
            return "DOUBLE";
        case matioCpp::ValueType::UTF8:
            return "UTF8";
        case matioCpp::ValueType::UTF16:
            return "UTF16";
        case matioCpp::ValueType::UTF32:
            return "UTF32";
        case matioCpp::ValueType::STRING:
            return "STRING";
        case matioCpp::ValueType::LOGICAL:
            return "LOGICAL";
        case matioCpp::ValueType::VARIABLE:
            return "VARIABLE";
        case matioCpp::ValueType::UNSUPPORTED:
            return "UNSUPPORTED";
    }
    return "UnknownValueType";
}

using matioCpp::File;

auto main(int argc, char* argv[]) -> int
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
            auto var = file.read(name); // flawfinder: ignore
            std::cout << "  " << name << ": " << variableTypeToString(var.variableType()) << " ("
                      << valueTypeToString(var.valueType()) << ") [";
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
