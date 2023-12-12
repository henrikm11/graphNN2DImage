//weight_initilization.hpp
//initializes weights using kaiming he initilization


#ifndef WEIGHT_INITILIZATION_HPP
#define WEIGHT_INITILIZATION_HPP

#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <iostream>
#include "tensor_template/tensor_template.h"
#include "neuralNetwork.h"

// recall typedef std::vector<std::vector<std::vector<size_t>>> slicedCoordTensor; 
template<size_t N>
void neuralNetwork<N>::biasInitilization_(void){
    std::random_device rd;
    std::mt19937 gen(rd());

    slicedCoordTensor biasCoord = biases_.shape.generateSlicedCoordinates();
    
    std::normal_distribution<double> distr(0.0, sqrt(2));
    for(size_t layer=1; layer<biasCoord.size(); layer++){
        for(const auto pos : biasCoord[layer]){
            double bias = distr(gen);
            biases_[layer].getEntry(pos)=bias;
        }
    }
    return;
}

template<size_t N>
void neuralNetwork<N>::weightInitilization_(void){
    std::random_device rd;
    std::mt19937 gen(rd());

    slicedCoordTensor weightCoord = weights_.shape.generateSlicedCoordinates();

    std::normal_distribution<double> distr(0.0, sqrt(2));
    for(const auto& pos : weightCoord[0]){
        double weight = distr(gen);
        weights_[0].getEntry(pos)=weight;  
    }
    for(size_t layer=1; layer<weightCoord.size(); layer++){
        size_t prevLayerCount = weights_[layer-1].shape.entryCount();
        double deviation = 2;
        deviation/=prevLayerCount;
        deviation=sqrt(deviation);
        std::normal_distribution<double> distr(0.0, deviation);
        for(const auto& pos : weightCoord[layer]){
            double weight = distr(gen);
            weights_[layer].getEntry(pos)=weight;
        }
    }
    return;
}









#endif /*WEIGHT_INITILIZATION_HPP*/