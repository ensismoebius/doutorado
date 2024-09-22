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

    // Desenhar a função seno dentro do painel
    this->content.width++;
    int limit = content.width > this->data.size() ? this->data.size() - 1 : content.width - 1;
    for (int i = 0; i < limit; i++)
    {
        startPosX = i + scroll.x - dimensions.width / 2;
        startPosY = this->data[i] + scroll.y;

        endPosX = startPosX + 1;
        endPosY = this->data[i + 1] + scroll.y;

        DrawLine(
            startPosX, 
            startPosY + this->yPlotOffset + dimensions.height / 2, 
            endPosX, 
            endPosY + this->yPlotOffset + dimensions.height / 2, 
            RED
            );
    }

    EndScissorMode(); // Fim da área de rolagem

    return this->isScrolling;
}