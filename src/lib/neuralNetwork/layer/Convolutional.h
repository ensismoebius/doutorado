#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

#include "../Config.h"
#include "../layer/Layer.h"

class Convolutional : public Layer
{
private:
public:
    Convolutional();
    ~Convolutional();
};
