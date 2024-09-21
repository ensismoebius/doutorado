#include "raylib.h"
#include "raygui.h"
#include <cmath>
#include <iostream>

int main()
{
    // Inicializa a janela
    InitWindow(800, 600, "GuiScrollPanel with Sine Plot");

    // Definir os parâmetros do painel de rolagem
    Rectangle dimensions = {20, 20, 400, 250}; // Tamanho do painel de rolagem visível
    Rectangle content = {20, 20, 400, 0};       // Tamanho do conteúdo do painel (maior para rolar)
    Vector2 scroll = {0, 0};                    // Posição de rolagem

    SetTargetFPS(60); // FPS desejado

    float scrollSpeed = 1.0f; // Velocidade de rolagem
    bool isScrolling = false;

    while (!WindowShouldClose())
    { // Loop principal
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Desenha o painel de rolagem
        isScrolling = GuiScrollPanel(dimensions, "", content, &scroll, nullptr);

        if (!isScrolling)
        {
            // Atualiza a posição de rolagem automaticamente
            scroll.x -= scrollSpeed;

            // Verifica se chegou ao final do conteúdo e reinicia a posição de rolagem
            if (scroll.x > content.width - dimensions.width)
            {
                scroll.x = 0;
            }
        }

        content.width++;

        // Desenha o conteúdo (gráfico de seno) dentro do painel
        // Desenhar senoide apenas na área visível
        BeginScissorMode(dimensions.x, dimensions.y, dimensions.width, dimensions.height);

        // Definir parâmetros do gráfico de seno
        float amplitude = 100; // Amplitude da senoide
        float frequency = .05;  // Frequência da senoide

        // DrawLine(int startPosX, int startPosY, int endPosX, int endPosY, Color color)
        int startPosX = 0;
        int startPosY = 0;
        int endPosX = 0;
        int endPosY = 0;

        // Desenhar a função seno dentro do painel
        for (int x = 0; x < content.width; x++)
        {
            float y = amplitude * sin(frequency * x) + amplitude + 50; // Função seno

            endPosX = x + scroll.x - dimensions.width / 2;
            endPosY = y + scroll.y;

            DrawLine(startPosX, startPosY, endPosX, endPosY, RED);

            startPosX = endPosX;
            startPosY = endPosY;
        }

        EndScissorMode(); // Fim da área de rolagem

        EndDrawing();
    }

    CloseWindow(); // Fecha a janela

    return 0;
}
