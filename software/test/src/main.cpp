#include "raylib.h"
#include "raygui.h"
#include "ScrollPanel.h"
#include <random>
int main()
{
    // Inicializa a janela
    InitWindow(800, 600, "GuiScrollPanel with Sine Plot");

    // Definir os parâmetros do painel de rolagem
    Rectangle dimensions = {20, 20, 600, 300}; // Tamanho do painel de rolagem visível

    ScrollPanel sp("Senoide", dimensions, RED);
    sp.autoScroll = true;
    sp.scrollSpeed = 1;
    sp.yPlotOffset = 50;

    std::random_device rd;                            // Usado para obter um valor aleatório a partir do hardware
    std::mt19937 gen(rd());                           // Inicializa o gerador com um valor aleatório
    std::uniform_int_distribution<short> dis(0, 255); // Intervalo para números aleatórios

    std::vector<short> data2(1024);
    std::vector<short> data(1024);

    for (auto &num : data)
    {
        num = dis(gen); // Preenche com números aleatórios
    }

    for (auto &num : data2)
    {
        num = dis(gen); // Preenche com números aleatórios
    }
    // SetTargetFPS(24);


    while (!WindowShouldClose())
    {
        ClearBackground(RAYWHITE);

        BeginDrawing();

        data2.insert(data2.end(), data.begin(), data.end());

        sp.draw(&data2);
        sp.handleInput();

        EndDrawing();
    }

    CloseWindow(); // Fecha a janela

    return 0;
}
