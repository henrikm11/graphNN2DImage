//backpropagation.hpp


#ifndef BACKPROPAGATION_HPP
#define BACKPROPAGATION_HPP

#include <vector>
#include "tensor_template/tensor_template.h"
#include "neuralNetwork.h"

/*
// TODO
// replace some access operations using Tensor getSlice operator if implementeted
// update gradient computation to include normalization if batches are being used
*/


//forward propagation update in layer at depth i
template<size_t N>
void neuralNetwork<N>::updateLayer_(const size_t i){
    assert(i>0);
    assert(i<neuronStates_.shape.size()); //if we call from updateNeuronStates_ we have thrown before
    
    for(const auto& currCoord : coordinates_[i]){
        //update neuron at these coordinates
        double updatedEntry=0;
        for(const auto& prevCoord : coordinates_[i-1]){
            //take input from neuron at these coordinates
            std::vector<size_t> weightCoord = {(size_t)(i-1)};
            weightCoord.insert(weightCoord.end(),prevCoord.begin(),prevCoord.end());
            weightCoord.insert(weightCoord.end(),currCoord.begin(),currCoord.end());
            double weight = weights_.getEntry(weightCoord);
            updatedEntry+=(weight*activationFct_(neuronStates_[i-1].getEntry(prevCoord)));
        }
        //updatedEntry=this->activationFct_(updatedEntry);
        //new version stores values of neurons before applying activation to it
        neuronStates_[i].getEntry(currCoord)=updatedEntry;
    }
    return;
}

//forward propagation through entire network
template<size_t N>
void neuralNetwork<N>::updateNeuronStates_(const Tensor<double,N-1>& input){
    assert(networkShape_.size()>0); //everything is pointless for an empty network...
    if(input.shape!=neuronStates_.shape[0]){
        throw(std::domain_error("neuralNetwork<N>::updateNeuronStates_: input shape does not match network shape"));
    }
    neuronStates_[0]=input;
    for(size_t i=1; i<neuronStates_.size(); i++){
        updateLayer_(i);
    }
    return;
}

//predict
template<size_t N>
Tensor<double,N-1> neuralNetwork<N>::predict(
    const Tensor<double,N-1>& input,
    bool updateNeuronStates
){
    if(updateNeuronStates){
        updateNeuronStates_(input);
    }
    Tensor<double,N-1> pred(neuronStates_.shape.back());
    for(const auto& c : coordinates_.back()){
        pred.getEntry(c)=activationFct_(neuronStates_[neuronStates_.size()-1].getEntry(c));
    }
}


//backpropagation all, but output layer


//update Gradient of weights and biases given a sample
//bool batch indicates is sample is part of a batch if so gradients are not overwritten but added to existing
template<size_t N>
void neuralNetwork<N>::updateGradient_(
    const Tensor<double, N-1>& sampleIn,
    const Tensor<double,N-1>& sampleOut,
    size_t batchSize
){
    /*
    call:
    forward propagation
    gradient updates for:
    final layer
    backpropagation
    weights
    biases
    */
    updateNeuronStates_(sampleIn);
    updateOutputLayerGrad_(sampleIn, sampleOut);
    backpropagation(sampleIn, sampleOut);
    updateWeightsGrad_(batchSize);
    updateBiasesGrad_(batchSize);
    return;
}

template<size_t N>
void neuralNetwork<N>::updateOutputLayerGrad_(
    const Tensor<double, N-1>& sampleIn,
    const Tensor<double,N-1>& sampleOut
){
    //call from updateGradient_ which updates neuronStates before
    Tensor<double,N-1> pred = predict(sampleIn, false);
    for(const auto& c : coordinates_[neuronStates_.size()-1]){
        neuronGrad_[neuronStates_.size()-1].getEntry(c)=
            costFctGrad_(pred, sampleOut, c, neuronCount_)
            *activationFctGrad_(neuronStates_[neuronStates_.size()-1].getEntry(c))
        ;
    }
    return;
}

//compute gradient in neuron state variables for all hidden and input layers
template<size_t N>
void neuralNetwork<N>::backpropagation_()
{
    for(size_t layer = coordinates_.size()-2; layer>-1; layer--){
        for(const auto& c : coordinates_[layer]){
            for(const auto& d : coordinates_[layer+1]){
                std::vector<size_t> weightCoord = {layer};
                weightCoord.insert(weightCoord.end(),c.begin(),c.end());
                weightCoord.insert(weightCoord.end(),d.begin(),d.end());
                neuronGrad_[layer].getEntry(c)+=
                    neuronGrad_[layer+1].getEntry(d)*weights_.getEntry(weightCoord)*activationFctGrad_(neuronStates_[layer].getEntry(c));
            }
        }
    }
    return;
}

//updates all weights assuming that neuronGrad_ has been updated
template<size_t N>
void neuralNetwork<N>::updateWeightsGrad_(size_t batchSize){
    for(size_t layer=0; layer<neuronStates_.size()-1; layer++){
        for(const auto& outCoord : coordinates_[layer]){
            //update weight going out of outCoord
            double updatedEntry=0;
            for(const auto& inCoord : coordinates_[layer+1]){
                //update weight connecting to iunCoord
                std::vector<size_t> weightCoord = {(size_t)(layer)};
                weightCoord.insert(weightCoord.end(),outCoord.begin(),outCoord.end());
                weightCoord.insert(weightCoord.end(),inCoord.begin(),inCoord.end());
                if(batchSize==1){
                    weightsGrad_.getEntry(weightCoord)=0;
                }
                weightsGrad_.getEntry(weightCoord)+=
                    neuronGrad_[layer].getEntry(inCoord)
                    *weights_.getEntry(weightCoord)
                ;
            }
        }
    }
    return;
}

       
       
    
//updates all biases, assume neuronGrad_ has been updated
template<size_t N>
void neuralNetwork<N>::updateBiasesGrad_(size_t batchSize){
    for(size_t layer=1; layer<neuronStates_.size(); layer++){
        for(const auto& c : coordinates_[layer]){
            if(batchSize==1){
                biasesGrad_[layer].getEntry(c)=0;
            }
            biasesGrad_[layer].getEntry(c)+=neuronGrad_[layer].getEntry(c);
        }
    }
    return;
}


#endif /*BACKPROPAGATION_HPP*/


