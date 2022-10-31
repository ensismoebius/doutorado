#include "experiment00.h"
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <armadillo>
#include <iostream>
#include <vector>

#include <Heatmap.h>

////////////////////////////////////////////
///////////////// SETTINGS /////////////////
////////////////////////////////////////////
const int SQUARE_SIZE = 50;
const float WINDOW_WIDTH = 800;
const float WINDOW_HEIGHT = 800;
const float TRAINING_RATE = 0.001;
const char FONT_PATH[] = "./res/fonts/Arial.ttf";
////////////////////////////////////////////
////////////// SETTINGS - END //////////////
////////////////////////////////////////////

std::vector<std::vector<float>> data0;
std::vector<std::vector<float>> data1;

void setData(std::vector<std::vector<float>>& data, unsigned amount)
{
    // data = {
    //     {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

    data.resize(amount);

    for (auto&& line : data) {
        line.resize(amount);

        for (auto&& item : line) {
            item = std::rand() / (float)RAND_MAX;
        }
    }
}

void experiment00()
{
    /////////////////////////////
    // Window creation - BEGIN //
    /////////////////////////////

    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    int32_t windowStyle = sf::Style::Default;

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Neural Network - float", windowStyle, settings);
    window.setFramerateLimit(0);

    // Puts the orign at the center
    sf::Vector2u size = window.getSize();
    sf::View view(sf::Vector2f(size.x / 2, size.y / 2), sf::Vector2f(WINDOW_WIDTH, WINDOW_HEIGHT));
    window.setView(view);

    ///////////////////////////
    // Window creation - END //
    ///////////////////////////

    // Ensuring that the random numbers are going to honor its names
    std::srand(std::time(nullptr));

    sf::Font font;
    font.loadFromFile(FONT_PATH);

    setData(data0, 8);
    sfplot::Heatmap hm0(font, 10, 1, 200, 200, 50, 50, data0);
    sfplot::Heatmap hm1(font, 10, 1, 200, 200, 50, 300, data0);
    hm0.axis = false;
    hm1.axis = false;

    auto rng = std::default_random_engine {};
    // Aplication main loop
    while (window.isOpen()) {
        window.clear(sf::Color::Magenta);

        setData(data0, 200);
        window.draw(hm0);
        window.draw(hm1);
        window.display();

        // Window and keyboard events
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Q)
                window.close();
        }
    }
}
