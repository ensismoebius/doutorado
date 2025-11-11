/*
 * Main entry point for the monografia C++ project
 * This project implements various neural network architectures and training
 * pipelines.
 */
#include <iostream>

#include "dataLoaders/MatFileUtils.h"

using matioCpp::utils::list_variable_names;
using matioCpp::utils::load_named_variable_as_matrix;
using std::cout;

auto main() -> int
{
    cout << "Monografia C++ Project" << '\n';
    cout << "Based on the dissertation: 'Autenticação Biométrica de Locutores "
            "Drasticamente Disfônicos Aprimorada pela Imagined Speech'"
         << '\n';

    auto var_names = list_variable_names(
        "/home/ensismoebius/Documentos/UNESP/"
        "doutorado/databases/BasedeDatosHablaImaginada/S02/"
        "S02_Audio.mat");
    for (const auto& name : var_names)
    {
        cout << "Found variable: " << name << '\n';
    }

    auto mat = load_named_variable_as_matrix(
        "/home/ensismoebius/Documentos/UNESP/"
        "doutorado/databases/BasedeDatosHablaImaginada/S02/"
        "S02_Audio.mat",
        "Audio");

    if (mat)
    {
        cout << "Loaded matrix with shape: " << mat->rows() << " x " << mat->cols() << '\n';
    }
    else
    {
        cout << "Failed to load matrix from MAT file.\n";
    }

    return 0;
}