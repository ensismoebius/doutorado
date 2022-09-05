#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <vector>

#include "../Config.h"
#include "../layer/Layer.h"

class Pooling : public Layer
{
private:
public:
    Pooling();
    ~Pooling();
};