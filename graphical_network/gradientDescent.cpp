//gradientDescent.cpp

#include "tensor_template.h"
#include "graphNeural.h"
#include "helper_fcts.h"


///@brief updates weight tensor on scale of learningRate in direction of gradient
template<class T, size_t M>
inline void gradientDescentStep( 
    Tensor<T,M>& weightsStates, 
    const Tensor<T,M>& weightsGradient,
    const double learningRate
    ){
        for(size_t i=0; i<weightsStates.size(); i++){
            //call gradient Descent on each weight
            gradientDescentStep(
                weightsStates[i],
                weightsGradient[i],
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
        for(size_t i=0; i<weightsState.size(); i++){
            weightsState[i]+=(learningRate*weightsGradient[i]);
        }
    
    return;
}


