#include "ScrollPanel.h"
#include <iostream>
#include <algorithm>

ScrollPanel::ScrollPanel(std::string title, Rectangle dimensions, Color color)
    : dimensions(dimensions),
      title(title),
      color(color)
{
    this->content = {dimensions.x, dimensions.y, dimensions.width, dimensions.height - 40};
}

void ScrollPanel::handleInput()
{
    // Verifica se o painel está em foco (mouse sobre ele)
    Vector2 mousePosition = GetMousePosition();
    bool isInFocus = CheckCollisionPointRec(mousePosition, this->dimensions);

    // Somente permitir zoom se o painel estiver em foco
    if (isInFocus)
    {
        // Aumentar ou diminuir o zoom com o scroll do mouse
        float mouseWheelMove = GetMouseWheelMove();
        if (mouseWheelMove != 0)
        {
            zoomLevel += mouseWheelMove * 0.1f;    // Ajuste sensibilidade do zoom com base no scroll
            zoomLevel = std::max(0.1f, zoomLevel); // Evita valores de zoom negativos ou muito pequenos
        }

        // Também permitir controle com as teclas UP e DOWN
        if (IsKeyPressed(KEY_UP))
        {
            zoomLevel += 0.1f;
        }

        if (IsKeyPressed(KEY_DOWN))
        {
            zoomLevel = std::max(0.1f, zoomLevel - 0.1f);
        }
    }
}

bool ScrollPanel::draw(std::vector<short> *data)
{
    // Desenha o painel de rolagem
    this->isScrolling = GuiScrollPanel(
        this->dimensions,
        this->title.c_str(),
        this->content,
        &this->scroll,
        nullptr);

    // Se o autoscroll estiver ativado e o usuário não estiver rolando manualmente
    if (autoScroll && !isScrolling)
    {
        // Ajusta a velocidade de rolagem proporcional ao número de dados e ao zoom
        scrollSpeed = 10 * this->zoomLevel;

        // Atualiza a posição de rolagem automaticamente
        this->scroll.x -= scrollSpeed;

        // Verifica se chegou ao final do conteúdo e reinicia a posição de rolagem
        if (this->scroll.x > content.width - dimensions.width)
        {
            this->scroll.x = 0;
        }
    }

    // Desenha o gráfico dentro do painel
    BeginScissorMode(dimensions.x, dimensions.y, dimensions.width, dimensions.height);

    // Aumenta o tamanho da área do gráfico se necessário
    if (this->content.width < data->size())
    {
        this->content.width++;
    }

    // Calcula o número de pontos visíveis, evitando desenhar fora da tela
    int startIdx = std::max(0, static_cast<int>(-scroll.x));                                    // Ponto inicial visível
    int endIdx = std::min(static_cast<int>(data->size() - 1), static_cast<int>(content.width)); // Ponto final visível

    // Plota os dados visíveis
    for (int i = startIdx; i < endIdx; i++)
    {
        startPosX = (i + scroll.x - dimensions.width / 2) * this->zoomLevel;
        startPosY = (data->at(i) + scroll.y) * this->zoomLevel;

        endPosX = startPosX + 1;
        endPosY = (data->at(i + 1) + scroll.y) * this->zoomLevel;

        DrawLine(
            startPosX,
            startPosY + this->yPlotOffset,
            endPosX,
            endPosY + this->yPlotOffset,
            color);
    }

    EndScissorMode(); // Fim da área de rolagem

    return this->isScrolling;
}