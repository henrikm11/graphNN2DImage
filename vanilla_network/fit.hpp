//fit.hpp

#include <vector>
#include <exception>
#include <cmath>
#include <random>
#include "tensor_template/tensor_template.h"
#include "neuralNetwork.h"
#include "weight_initilization.hpp"


/// @brief returns a vector of random integers, can have repetitions
/// @param upperBound range for random numbers is [0,upperBound)
/// @param count number of random numbers pickes
/// @return vector of random numbers
std::vector<size_t> getRandomIdx(size_t upperBound, size_t count){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distr(0,upperBound-1);

    std::vector<size_t> randomIdx;
    for(size_t i=0; i<count; i++){
        randomIdx.push_back(distr(gen));
    }
    return randomIdx;
}


template<size_t N>
void neuralNetwork<N>::fit_(
    double learningRate,
    size_t maxEpochs,
    size_t batchSize
){
    weightInitilization_();
    size_t epochCount=0;
    for(size_t epochCount=0; epochCount<maxEpochs; epochCount++){
        size_t batchCount = trainingIns_.size()/batchSize+1; //number of batches in one epoch
        for(size_t batch=0; batch<batchCount; batch++){
            std::vector<size_t> randomIdx = getRandomIdx(trainingIns_.size(),batchSize);
            size_t batchIdx=0;
            for(size_t i : randomIdx){
                Tensor<double,N-1> sampleIn = trainingIns_[i];
                Tensor<double,N-1> sampleOut = trainingOuts_[i];
                updateGradient_(
                    sampleIn,
                    sampleOut,
                    batchIdx
                );
                netGradientDescentStep(learningRate);
                batchIdx++;
            }
        }
    }
    return;
}


template<size_t N>
void neuralNetwork<N>::fit(
    const std::vector<Tensor<double,N-1>>& trainingIns,
    const std::vector<Tensor<double,N-1>>& trainingOuts,
    double learningRate,
    size_t maxEpochs,
    size_t batchSize
)
{
    //check if training data matches shapes
    for(const auto& sampleIn : trainingIns){
        if(sampleIn.shape!=networkShape_[0].shape){
            throw(std::domain_error("neuralNetwork<N>::fit: shape of inputs does not match networkShape"));
        }
    }
    for(const auto& sampleOut : trainingOuts){
        if(sampleOut.shape!=networkShape_[networkShape_.size()-1].shape){
            throw(std::domain_error("neuralNetwork<N>::fit: shape of outputs does not match networkShape"));
        }
    }

    //copy data into class
    //do i want this?
    trainingIns_=trainingIns;
    trainingOuts_=trainingOuts;

    fit_(
        learningRate,
        maxEpochs,
        batchSize
    );

    return;
}

