//tensor_template.h

#include <vector>

#ifndef TENSOR_TEMPLATE_H
#define TENSOR_TEMPLATE_H


//tensor shape template

template<size_t N>
struct TensorShape{
    //TensorShape(); //default constructor, constructs TensorShape of size 0
    TensorShape(const std::vector<TensorShape<N-1>>& shape);
    //copy and destructor are default
    //assignment deleted because of const member variable
    const TensorShape<N-1>& operator[](const int i) const;
    size_t size(void) const;
    const std::vector<TensorShape<N-1>> shape_;
};


template<>
//a 1 tensor is a vector so this is just a const int
struct TensorShape<1>{
    TensorShape(const unsigned int size=0):shape_(size){};
    //copy and destructor are default
    //assignment deleted because of const member variable
    size_t size(void)const {return shape_;}
    const unsigned int shape_;
};


//declaration and immplementation of comparison operators 
//operator==(const TensorShape<M> lhs&, const TensorShape<N> rhs&)
//operator!=(const TensorShape<M> lhs&, const TensorShape<N> rhs&)
//are in tensor_template.tpp



template<typename T, size_t N>
class Tensor{
public:
    Tensor(const std::vector<Tensor<T,N-1>>& entries);
    //copy and destructor are default
    //assignment deleted because of const member variable
    Tensor<T,N>& operator=(const Tensor<T,N>& other);
    //assignment throws if shapes don't match, otherwise default


    size_t size(void) const;
    Tensor<T,N-1>& operator[](const int i);
    const Tensor<T,N-1>& at(const int i) const;
    const TensorShape<N>& shape(void) const;

private:
    const TensorShape<N> shape_;
    std::vector<Tensor<T,N-1>> entries_;
};


template<typename T>
class Tensor<T,1>{
public:
    //default default constructor is ok
    Tensor(const std::vector<T>& entries);
    Tensor(int shape);
    Tensor(TensorShape<1> shape);
    Tensor<T,1>& operator=(const Tensor<T,1>& other);
    //copy and destructor are default
    //assignment deleted because of const member variable
    size_t size(void) const;
    T& operator[](const int i);
    const T& at(const int i) const;
    const TensorShape<1>& shape(void);

private:
    const TensorShape<1> shape_; //just an int
    std::vector<T> entries_;
};


//helper function to get shape from vector of N-1 tensors
template<typename T, size_t N>
TensorShape<N> getShape(const std::vector<Tensor<T,N-1>>& entries);

template<typename T>
TensorShape<1> getShape(const Tensor<T,1>& tensor);



#include "tensor_template.tpp"
#endif /*TENSROR_TEMPLATE_H*/