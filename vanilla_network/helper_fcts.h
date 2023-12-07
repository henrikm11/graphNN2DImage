//helper_fcts.h


/*
//various helper functions
//eg activation functions, their gradients etc
//typically inlined
*/

/*
//TO DO
//pass mse and mse grad coordinates through pointer?
//no need to generate them over and over again
// same for gradient
//alternative compute it iteratively using the following
// given x_1,..x_n and their mean \mu_1
// given x_n+1,..x_n+m and their mean \mu_2
// the meain of x_1,..,x_n+m is
// n/(n+m) \mu_1 + m/(n+m) \mu_2

-) error handling in mse grad?
*/

#ifndef HELPER_FCTS_H
#define HELPER_FCTS_H

#include <cmath>
#include <vector>
#include <exception>
#include "tensor_template/tensor_template.h"


//struct containing a function and its gradient
struct fctGrad{
    fctGrad(
        double (* const fct)(double),
        double (* const grad)(double)
    )
    :fct(fct),
    grad(grad)
    {
        if(fct==nullptr){
            throw(std::invalid_argument("fctGrad(): function must be non null"));
        }
         if(grad==nullptr){
            throw(std::invalid_argument("fctGrad(): gradient function must be non null"));
        }
        return;
    }
    double (* const fct)(double);
    double (* const grad)(double);
};



//relu activation

inline double relu(double x){
    return std::max(x,(double)0);
}
inline double reluGrad(double x){
    return (double)(x>0);
}

//mean squared error for vectors
double inline mse(const std::vector<double>& a, const std::vector<double>& b){
    if(a.size()!=b.size()){
        throw(std::domain_error("mse: vector sizes are different"));
    }

    double currErr=0;
    double prevErr=0;
    double count=1;

    auto it_a = a.begin();
    auto it_b = b.begin();

    while(it_a!=a.end()){
        double temp = currErr;
        double currCont=(*it_a-*it_b)*(*it_a-*it_b); //square error of current term
        currErr+=(currCont-prevErr)/count;
        prevErr=temp;
        count++;
    }
    return currErr;
}

//computes derivative of mse(a,b) in direction pos in first (a) component
double inline mseGrad(const std::vector<double>& a, const std::vector<double>& b, size_t pos){
    if(a.size()!=b.size()){
        throw(std::domain_error("mse: vector sizes are different"));
    }
    if(a.size()==0){
        return 0;
    }
    //if pos is bad we are throwing anyway
    return 2*(a[pos]-b[pos])/a.size();



}

//mean squared error for tensors
//computes mse of tensors with matching shape,
//uses nummerically stable formula for mean \mu_n of numbers x_1,...,x_n:
// \mu_n=\mu_{n-1}+(x_n-\mu_{n-1})
template<size_t N>
double mse(Tensor<double,N>& a, Tensor<double,N>& b){
    if(a.shape!=b.shape){
        throw(std::domain_error("mse: tensor shapes are different"));
    }
    std::vector<std::vector<size_t>> coord = a.shape.generateCoordinates();
    double currErr=0;
    double prevErr=0;
    double count=1;

    for(const auto& c : coord){
        double temp=currErr;
        double currCont=(a.getEntry(c)-b.getEntry(c))*(a.getEntry(c)-b.getEntry(c)); //square error of current term
        currErr+=(currCont-prevErr)/count;
        prevErr=temp;
        count++;
    }
    return currErr;
}

//mse gradient wrt to first entries of first tensor
//this throws std::out_of_range if pos is no good
//entryCount is total count of entries of tensors, no need to compute this at every pass
template<size_t N>
double mseGrad(Tensor<double,N>& a, Tensor<double,N>& b, const std::vector<size_t>& pos, size_t entryCount){
    if(a.shape!=b.shape){
        throw(std::domain_error("mseGrad: tensor shapes are different"));
    }

    double grad = 2*(a.getEntry(pos)-b.getEntry(pos))/entryCount;
    return grad;
}



//crossEntropy
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