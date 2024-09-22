#include "ScrollPanel.h"

ScrollPanel::ScrollPanel(std::string title, Rectangle dimensions, Color color)
    : dimensions(dimensions),
      title(title),
      color(color)
{
    this->content = {dimensions.x, dimensions.y, dimensions.width, dimensions.height - 40};
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

    // Desenha o gráfico dentro do painel
    BeginScissorMode(dimensions.x, dimensions.y, dimensions.width, dimensions.height);

    // Aumenta o tamanho da área do gráfico se necessário
    this->content.width++;

    // Calcula o número de pontos visíveis, evitando desenhar fora da tela
    int startIdx = std::max(0, static_cast<int>(-scroll.x));                                         // Ponto inicial visível
    int endIdx = std::min(static_cast<int>(this->data.size() - 1), static_cast<int>(content.width)); // Ponto final visível

    // Plota os dados visíveis
    for (int i = startIdx; i < endIdx; i++)
    {
        startPosX = i + scroll.x - dimensions.width / 2;
        startPosY = this->data[i] + scroll.y;

        endPosX = startPosX + 1;
        endPosY = this->data[i + 1] + scroll.y;

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