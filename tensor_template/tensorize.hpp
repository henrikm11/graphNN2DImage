//tensorize.hpp

#ifndef TENSORIZE_HPP
#define TENSORIZE_HPP

#include <vector>
#include <exception>
#include "tensor_template.h"



template<typename T, size_t N>
T applyToTensor(const Tensor<T,N>& tensor1, const Tensor<T,N>& tensor2, T (*pointwiseFct)(const T, const T)){
    if(tensor1.shape!=tensor2.shape){
        throw(std::domain_error("T apply(Tensor<T,N>,Tensor<T,N>, fct): tensor shapes do not match"));
    }
    T res(0);
    for(size_t i=0; i<tensor1.size(); i++){
        res+=applyToTensor(tensor1.at(i),tensor2.at(i), pointwiseFct);
    }
    return res;

}


template<typename T>
T applyToTensor(const Tensor<T,1>& tensor1, const Tensor<T,1>& tensor2, T (*pointwiseFct)(const T, const T)){
    if(tensor1.size()!=tensor2.size()){
        throw(std::domain_error("T apply(Tensor<T,1>,Tensor<T,1>, fct): tensor shapes do not match"));
    }
    T res(0); //we assume that T has a constructor T(int)
    for(size_t i=0; i<tensor1.size(); i++){
        res+=pointwiseFct(tensor1.at(i), tensor2.at(i));
    }
    return res;
}





#endif /*TENSORIZE_HPP*/


