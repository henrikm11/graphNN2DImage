//helper_fcts.h


#ifndef HELPER_FCTS_H
#define HELPER_FCTS_H

#include "graphNeural.h"
#include <vector>

/// @brief returns 8 directional neighborhood
/// @return list of neighbors as pairs
std::vector<std::pair<int,int>> getNeighbors(int vertSize, int horSize, int vertPos, int horPos);




//inline functions

//activation function
/// @brief computes cross entropy of distribution probabilities and indicator at correct label 
/// @param probabilities predicted probabilities
/// @param correctLabel correct label, labels are in range 0,1,...
/// @return cross entropy 
inline double crossEntropy(std::vector<double> probabilities, int correctLabel){
    return -std::log(probabilities[correctLabel]);
}

inline std::vector<double> crossEntropyGradient(std::vector<double> probabilities, int correctLabel){
    std::vector<double> grad(probabilities.size(),0);
    grad[correctLabel]=-1/probabilities[correctLabel];
    return grad;
}

inline double sigma(const double input){
    //ReLU activation function
    return (input>0)*input;
}

inline double sigmaPrime(const double input){
    //derivative of ReLU
    return (input>=0);
}

//some smooth approximations of argmax, max, its gradient

/// @brief smmoth approximation of argmax
/// @param input vector to take soft argmax of
/// @param beta regularization parameter, get argmax approached for beta \to \infty
/// @return vector approximating 1_argmax
inline std::vector<double> softArgMax(const std::vector<double>& input, double beta=10){
    std::vector<double> res(input.size(),0);

    double normalization=0;
    for(double coordinate : input){
        normalization+=std::exp(beta*coordinate);
    }

    for(size_t i=0; i<input.size(); i++){
        double coordinate=input[i];
        res[i]=std::exp(beta*coordinate)/normalization;
    }

    return res;
}

inline std::vector<std::vector<double>> softArgMaxGrad(const std::vector<double>& input, double beta=10){
    std::vector<std::vector<double>> res(input.size(), std::vector<double>(input.size(),0));
    //res[i][j]=\partial softMax_i / \partial x_j
    std::vector<double> f = softArgMax(input, beta);
    for(int i=0; i<input.size(); i++){
        for(int j=0; j<input.size(); j++){
            res[i][j]=beta*f[i]*((i==j) - f[j]);
        }
    }
    return res;
}

inline std::vector<double> softArgMaxGradComponent(const std:: vector<double>& input, int idx, double beta=10){
    std::vector<double> res(input.size(),0);
    //res[j]=\partial softMax_idx / \partial x_j
    std::vector<double> f = softArgMax(input, beta);
    for(int j=0; j<input.size(); j++){
        res[j]=beta*f[idx]*((idx==j) - f[j]);
    }
    return res;
}

/// @brief smooth approximation of max
/// @param input vector to take max of  
/// @param alpha regularization parameter, get max for \alpha \to \infty
/// @return double approximating max of input vector
inline double BoltzmannOperator(const std::vector<double>& input, double alpha=10){
    double res=0;
    double normalization=0;
    for(double coordinate : input){
        normalization+=std::exp(alpha*coordinate);
    }
    for(double coordinate : input){
        double contribution=coordinate*std::exp(alpha*coordinate)/normalization;
        res+=contribution;
    }
    return res;
}

/// @brief computes gradient vector of smooth approximation of max
/// @param input vector at which we evaluate gradient
/// @param alpha regularization parameter
/// @return gradient vector at input
inline std::vector<double> BoltzmannOperatorGrad(const std::vector<double>& input, double alpha=10){
    std::vector<double> gradient(input.size(),0);

    double BoltzmannOp=BoltzmannOperator(input,alpha);

    double normalization=0;
    for(double coordinate : input){
        normalization+=std::exp(alpha*coordinate);
    }

    std::vector<double> factors = softArgMax(input, alpha);

    for(size_t i=0; i<input.size(); i++){
        double coordinate=input[i];
        gradient[i]=factors[i]*(1+alpha*(coordinate-BoltzmannOp));
    }

    return gradient;
}

#endif //HELPER_FCTS_H
