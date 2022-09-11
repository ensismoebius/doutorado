#include <armadillo>

#define m arma::Mat<float>

m linreg(m x, m w, m b) {
    return arma::dot(x,w) + b;
}

void experiment01(){
  
}