#include "MatFile.hpp"

#include <memory>
#include <optional>
#include <stdexcept>

MatFile::MatFile(const std::string& filename)
{
    matfp_ = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (matfp_ == nullptr)
    {
        throw std::runtime_error("Error opening MAT file: " + filename);
    }
}

MatFile::~MatFile()
{
    if (matfp_ != nullptr)
    {
        Mat_Close(matfp_);
    }
}

auto MatFile::getVariableNames() -> std::vector<std::string>
{
    std::vector<std::string> names;
    matvar_t* matvar = nullptr;
    while ((matvar = Mat_VarReadNextInfo(matfp_)) != nullptr)
    {
        names.emplace_back(matvar->name);
        Mat_VarFree(matvar);
    }
    return names;
}

auto MatFile::readVariable(const std::string& varName) -> matvar_t*
{
    return Mat_VarRead(matfp_, varName.c_str());
}

auto MatFile::readFirstNumericVariable()
    -> std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>>
{
    if (matfp_ == nullptr)
    {
        return std::nullopt;
    }

    matvar_t* var = nullptr;
    while ((var = Mat_VarReadNext(matfp_)) != nullptr)
    {
        // For now, we only consider double-precision floating-point matrices as numeric
        // You might want to extend this to other numeric types (e.g., MAT_C_SINGLE, MAT_C_INT32)
        if (var->class_type == MAT_C_DOUBLE)
        {
            return std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>>{
                std::unique_ptr<matvar_t, void (*)(matvar_t*)>(var, &Mat_VarFree)};
        }
        Mat_VarFree(var); // Free non-numeric variables
    }

    return std::nullopt;
}
