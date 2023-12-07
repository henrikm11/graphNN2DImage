//neuralNetwork.hpp

#ifndef NEURAL_NETWORK_TPP
#define NEURAL_NETWORK_TPP


#include "tensor_template/tensor_template.h"
#include "neuralNetwork.h"
#include <vector>
#include <stdexcept>


template<size_t N>
TensorShape<2*N-1> getWeightsShape(const TensorShape<N>& networkShape){
    std::vector<TensorShape<2*N-2>> slices;
    for(size_t i=0; i<networkShape.size()-1;i++){
        slices.emplace_back(concatShapes(networkShape[i],networkShape[i+1]));
    }
    TensorShape<2*N-1> shape(slices);
    return shape;
}


template<>
TensorShape<1> getWeightsShape(const TensorShape<1>& networkShape){
    TensorShape<1> shape(networkShape.size()-1);
    return shape;
}



template<size_t N>
neuralNetwork<N>::neuralNetwork(
    const TensorShape<N>& shape,
    double (*const activationFunction)(double),
    double (*const activationFunctionGrad)(double),
    double (*const errorFct)(Tensor<double,N-1>&, Tensor<double, N-1>&),
    double (*const errorFctGrad)(Tensor<double,N-1>&, Tensor<double,N-1>&, const std::vector<size_t>&, size_t),
    double learningRate
    )
    :neuronCount_(shape.entryCount()),
    networkShape_(shape),
    activationFct_(activationFunction),
    activationFctGrad_(activationFunctionGrad),
    costFct_(errorFct),
    costFctGrad_(errorFctGrad),
    weights_(getWeightsShape(shape)),
    biases_(shape),
    weightsGrad_(getWeightsShape(shape)),
    biasesGrad_(shape),
    neuronStates_(shape),
    neuronGrad_(shape),
    learningRate_(learningRate),
    coordinates_(shape.generateSlicedCoordinates())
    {
        if(activationFct_==nullptr){
            throw(std::invalid_argument("neuralNetwork<N>::neuralNetwork(): activationFunction must be provided"));
        }
        if(activationFctGrad_==nullptr){
            throw(std::invalid_argument("neuralNetwork<N>::neuralNetwork(): activationFunctionGrad must be provided"));
        }
        return;
    }


template<size_t N>
void neuralNetwork<N>::setLearningRate(double rate){
    if(rate<=0){
        throw std::domain_error("neuralNetwork<N>::setLearningRate : learningRate needs to be positive");
    }
    learningRate_=rate;
    return;
}



#endif /* NEURAL_NETWORK_TPP */

