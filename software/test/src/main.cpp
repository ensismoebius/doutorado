#include "raylib.h"
#include "raygui.h"
#include "ScrollPanel.h"

#include <cmath>
#include <iostream>

int main()
{
    // Inicializa a janela
    InitWindow(800, 600, "GuiScrollPanel with Sine Plot");

    // Definir os parâmetros do painel de rolagem
    Rectangle dimensions = {20, 20, 600, 250}; // Tamanho do painel de rolagem visível

    ScrollPanel sp("Ola", dimensions);
    sp.autoScroll = true;
    sp.scrollSpeed = 1;

    SetTargetFPS(60); // FPS desejado

    while (!WindowShouldClose())
    { // Loop principal
        BeginDrawing();
        ClearBackground(RAYWHITE);

        sp.draw();

        EndDrawing();
    }

    CloseWindow(); // Fecha a janela

    return 0;
}
