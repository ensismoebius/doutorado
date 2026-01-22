#include <cstdlib>
#include <iostream>

extern "C"
{
    void* criar_modelo_snn(int num_inputs, int num_outputs, int profundidade);
    void destruir_modelo_snn(void*);
    char* rede_snn_descricao(void*);
}

int main()
{
    void* m = criar_modelo_snn(100, 10, 3);
    if (!m)
    {
        std::cerr << "failed to create model" << std::endl;
        return 1;
    }
    char* desc = rede_snn_descricao(m);
    if (desc)
    {
        std::cout << "desc: " << desc << std::endl;
        free(desc);
    }
    else
    {
        std::cout << "no desc" << std::endl;
    }
    destruir_modelo_snn(m);
    return 0;
}
