#pragma once

#include <vector>
#include <stdexcept>

#include "../Config.h"
#include "../layer/Layer.h"

class Pooling : public Layer
{
public:
    Pooling();
    ~Pooling();
};