/**
 * @ Author: Your name
 * @ Create Time: 2024-09-21 18:28:09
 * @ Modified by: Your name
 * @ Modified time: 2024-09-21 19:24:26
 * @ Description:
 */

#ifndef __SCROOLPANEL_H__
#define __SCROOLPANEL_H__

#include "raylib.h"
#include "raygui.h"
#include <cmath>
#include <string>

class ScrollPanel
{

private:
    // Definir os parâmetros do painel de rolagem
    Rectangle content = {0, 0, 400, 200};  // Tamanho do conteúdo do painel (maior para rolar)
    Rectangle dimensions = {0, 0, 400, 0}; // Tamanho do painel de rolagem visível
    Vector2 scroll = {0, 0};               // Posição de rolagem
    std::string title;

public:
    bool autoScroll = false;
    bool isScrolling = false;   // O usuário está rolando?
    float scrollSpeed = 1.0f; // Velocidade de rolagem

public:
    ScrollPanel(std::string title, Rectangle dimensions);

    bool draw();
};
#endif // __SCROOLPANEL_H__