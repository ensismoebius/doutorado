#ifndef __SCROOLPANEL_H__
#define __SCROOLPANEL_H__

#include "raylib.h"
#include "raygui.h"
#include <cmath>

class ScroolPanel
{

private:
    // Definir os parâmetros do painel de rolagem
    Rectangle content = {0, 0, 400, 200};  // Tamanho do conteúdo do painel (maior para rolar)
    Rectangle dimensions = {0, 0, 400, 0}; // Tamanho do painel de rolagem visível
    Vector2 scroll = {0, 0};               // Posição de rolagem
    char *title;

public:
    bool autoScroll = false;
    bool isScrolling = false;   // O usuário está rolando?
    float scrollSpeed = 100.0f; // Velocidade de rolagem

public:
    ScroolPanel(
        char *title,
        Rectangle dimensions) : dimensions(dimensions),
                                title(title)
    {
        this->content = {dimensions.x, dimensions.y, dimensions.width, 0}; // Tamanho do conteúdo do painel (maior para rolar)
    }

    bool draw()
    {
        // Desenha o painel de rolagem
        this->isScrolling = GuiScrollPanel(
            this->dimensions,
            this->title,
            this->content,
            &this->scroll,
            nullptr);

        // Ativa ou desativa o autoscroll se o usuárie está rolando
        if (autoScroll && !isScrolling)
        {
            // Atualiza a posição de rolagem automaticamente
            this->scroll.x -= scrollSpeed;

            // Verifica se chegou ao final do conteúdo e reinicia a posição de rolagem
            if (this->scroll.x > content.width - dimensions.width)
            {
                this->scroll.x = 0;
            }
        }

        this->content.width++;

        // Desenha o conteúdo (gráfico de seno) dentro do painel
        // Desenhar senoide apenas na área visível
        BeginScissorMode(dimensions.x, dimensions.y, dimensions.width, dimensions.height);

        // Definir parâmetros do gráfico de seno
        float amplitude = 100; // Amplitude da senoide
        float frequency = .05; // Frequência da senoide

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

        return this->isScrolling;
    }
}
#endif // __SCROOLPANEL_H__