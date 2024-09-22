#include "raylib.h"
#include "raygui.h"
#include "ScrollPanel.h"
int main()
{
    // Inicializa a janela
    InitWindow(800, 600, "GuiScrollPanel with Sine Plot");

    // Definir os parâmetros do painel de rolagem
    Rectangle dimensions = {20, 20, 600, 250}; // Tamanho do painel de rolagem visível

    ScrollPanel sp("Senoide", dimensions);
    sp.autoScroll = true;
    sp.scrollSpeed = 1;

    std::vector<short> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    while (!WindowShouldClose())
    {
        ClearBackground(RAYWHITE);

        SetTargetFPS(60);

        BeginDrawing();

        sp.draw(data);

        EndDrawing();
    }

    CloseWindow(); // Fecha a janela

    return 0;
}
