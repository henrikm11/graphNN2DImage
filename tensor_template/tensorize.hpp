//tensorize.hpp
//template to aggregate functions TxT->T for tensors


#ifndef TENSORIZE_HPP
#define TENSORIZE_HPP

#include <vector>
#include <exception>
#include "tensor_template.h"


template<typename T, size_t N>
Tensor<T,N> transform(const Tensor<T,N>& input, T (*const pointwiseTransform)(const T)){
    Tensor<T,N> res(input.shape);
    for(size_t i=0; i<input.size();i++){
        res[i]=transform(input.at(i),pointwiseTransform);
    }
    return res;
}

template<typename T>
Tensor<T,1> transform(const Tensor<T,1>& input, T (*const pointwiseTransform)(const T)){
    Tensor<T,1> res(input.shape);
    for(size_t i=0; i<input.size();i++){
        res[i]=pointwiseTransform(input.at(i));
    }
    return res;
}


template<typename T,size_t N>
void transformInPlace(const Tensor<T,N>& input, Tensor<T,N>& output, T (*const pointwiseTransform)(const T)){
    for(size_t i=0; i<input.size(); i++){
        transformInPlace(input.at(i),output[i],pointwiseTransform);
    }
    return;
}

template<typename T>
void transformInPlace(const Tensor<T,1>& input, Tensor<T,1>& output, T (*const pointwiseTransform)(const T)){
    for(size_t i=0; i<input.size(); i++){
        output[i]=pointwiseTransform(input.at(i));
    }
    return;
}


template<typename T, size_t N>
T aggregateOnTensor(const Tensor<T,N>& tensor1, const Tensor<T,N>& tensor2, T (*pointwiseFct)(const T, const T)){
    if(tensor1.shape!=tensor2.shape){
        throw(std::domain_error("T ggregateOnTensor(Tensor<T,N>,Tensor<T,N>, fct): tensor shapes do not match"));
    }
    T res(0);
    for(size_t i=0; i<tensor1.size(); i++){
        res+=aggregateOnTensor(tensor1.at(i),tensor2.at(i), pointwiseFct);
    }
    return res;

}


template<typename T>
T aggregateOnTensor(const Tensor<T,1>& tensor1, const Tensor<T,1>& tensor2, T (*pointwiseFct)(const T, const T)){
    if(tensor1.size()!=tensor2.size()){
        throw(std::domain_error("T ggregateOnTensor(Tensor<T,1>,Tensor<T,1>, fct): tensor shapes do not match"));
    }
    T res(0); //we assume that T has a constructor T(int)
    for(size_t i=0; i<tensor1.size(); i++){
        res+=pointwiseFct(tensor1.at(i), tensor2.at(i));
    }
    return res;
}





#endif /*TENSORIZE_HPP*/


