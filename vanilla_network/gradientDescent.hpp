//gradientDescent.hpp


#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "tensor_template/tensor_template.h"
#include "helper_fcts.h"
#include "neuralNetwork.h"



///@brief updates weight tensor on scale of learningRate in direction of gradient
template<class T, size_t M>
inline void gradientDescentStep( 
    Tensor<T,M>& weightsStates, 
    const Tensor<T,M>& weightsGradient,
    const double learningRate
    ){
        if(weightsStates.shape!=weightsGradient.shape){
            throw(std::domain_error("gradientDescentShape: size of function and gradient do not match"));
        }
        for(size_t i=0; i<weightsStates.size(); i++){
            //call gradient Descent on each weight
            gradientDescentStep(
                weightsStates[i],
                weightsGradient.at(i),
                learningRate
            );
        }
    return;
}

//template specialization to N=1
template<class T>
inline void gradientDescentStep(
    Tensor<T,1>& weightsState,
    const Tensor<T,1>& weightsGradient,
    const double learningRate
    ){
        if(weightsState.shape!=weightsGradient.shape){
            throw(std::domain_error("gradientDescentStep: size of function and gradient do not match"));
        }
        for(size_t i=0; i<weightsState.size(); i++){
            weightsState[i]-=(learningRate*weightsGradient.at(i)); //corrected sign
        }
        
    return;
}

template<size_t N>
void neuralNetwork<N>::netGradientDescentStep_(double learningRate){
    //weights
    gradientDescentStep( 
        weights_, 
        weightsGrad_,
        learningRate
    );
    //biases
    gradientDescentStep(
        biases_,
        biasesGrad_,
        learningRate
    );
    return;
}



#endif /*GRADIENT_DESCENT_HPP*/
