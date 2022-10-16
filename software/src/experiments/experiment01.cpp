#include <armadillo>

#define m arma::Mat<float>

m linreg(m x, m w, m b)
{
    return x * w + b;
}

void initializesWeights(m weights)
{
    weights.randu(weights.n_rows, weights.n_cols);
}

// Generate y = X w + b + noise
m syntheticData(m w, float b, int numExamples)
{
    m X(numExamples, w.n_rows);

    m y = linreg(X, w, {b});

    // Add noise
    m noise(y.n_rows, y.n_cols);
    noise.randu(y.n_rows, y.n_cols);

    y += noise;

    return y;
}

void experiment01()
{
    m w(1000, 2);
    m trueW({2, -3.4f});
    float trueB = 4.2f;
    m y = syntheticData(trueW, trueB, 1000);
}