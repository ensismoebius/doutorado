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

    std::vector<short> data(1024);
    std::random_device rd;                            // Usado para obter um valor aleatório a partir do hardware
    std::mt19937 gen(rd());                           // Inicializa o gerador com um valor aleatório
    std::uniform_int_distribution<short> dis(0, 255); // Intervalo para números aleatórios

    for (auto &num : data)
    {
        num = dis(gen); // Preenche com números aleatórios
    }
    // SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        ClearBackground(RAYWHITE);

        BeginDrawing();

        sp.draw(data);

        EndDrawing();
    }

    CloseWindow(); // Fecha a janela

    return 0;
}
