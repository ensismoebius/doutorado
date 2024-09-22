/**
 * @ Author: Your name
 * @ Create Time: 2024-09-21 18:28:09
 * @ Modified by: Your name
 * @ Modified time: 2024-09-21 23:05:35
 * @ Description:
 */

#ifndef __SCROOLPANEL_H__
#define __SCROOLPANEL_H__

#include "raylib.h"
#include "raygui.h"
#include <vector>
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
    std::vector<short> data;


    int x = 0;

    int startPosX = 0;
    int endPosX = 0;

    int startPosY = 0;
    int endPosY = 0;

public:
    bool autoScroll = false;
    bool isScrolling = false; // O usuário está rolando?
    float scrollSpeed = 1.0f; // Velocidade de rolagem

public:
    ScrollPanel(std::string title, Rectangle dimensions);

    bool draw(std::vector<short> data);
};
#endif // __SCROOLPANEL_H__