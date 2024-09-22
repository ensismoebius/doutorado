#include "ScrollPanel.h"
#include <iostream>
#include <cmath>

ScrollPanel::ScrollPanel(std::string title, Rectangle dimensions)
    : dimensions(dimensions),
      title(title)
{
    this->content = {dimensions.x, dimensions.y, dimensions.width, dimensions.height}; // Tamanho do conteúdo do painel (maior para rolar)
}

bool ScrollPanel::draw(std::vector<short> data)
{

    this->data.insert(this->data.end(), data.begin(), data.end());

    // Desenha o painel de rolagem
    this->isScrolling = GuiScrollPanel(
        this->dimensions,
        this->title.c_str(),
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

    // Desenha o conteúdo (gráfico de seno) dentro do painel
    // Desenhar senoide apenas na área visível
    BeginScissorMode(dimensions.x, dimensions.y, dimensions.width, dimensions.height);

    this->endPosX = 0;
    this->endPosY = 0;

    // Desenhar a função seno dentro do painel
    this->content.width++;

    int limit = content.width > this->data.size() ? this->data.size() : content.width;

    for (int i = 0; i < limit; i++)
    {
            float y = this->data[i] + 50; // Função seno

            endPosX = i + scroll.x - dimensions.width / 2;
            endPosY = y + scroll.y;

            DrawLine(startPosX, startPosY, endPosX, endPosY, RED);

            startPosX = endPosX;
            startPosY = endPosY;
    }

    EndScissorMode(); // Fim da área de rolagem

    return this->isScrolling;
}