//tensor_template.h

/*
template for 'inhomogeneous tensors'
a tensor of type <T,N> is given by a fixed(!) size std::vector of tensors of type <T,N-1>
a 1 tensor is nothing but a fixed(!) size std::vector<T>

natural framework for neural networks with varying number of neurons across layers
should not be used for N large because of potential memory fragmentation
fixed size does in principal allow to use a continuous block of memory, but allowing different sizes for different components makes the index arithmetics of this extremely complicated
benefit of using iterated vectors is that Tensor<T,N>::operator[] returns Tensor<T,N-1>& and not an artificial reference type object.
*/



/*
//TO DO
//-) move semantics would be nice to have
*/

#ifndef TENSOR_TEMPLATE_H
#define TENSOR_TEMPLATE_H


#include <vector>


//tensor shape template

template<size_t N>
struct TensorShape{
    //TensorShape(); //default constructor, constructs TensorShape of size 0
    TensorShape(const std::vector<TensorShape<N-1>>& shape);
    TensorShape(size_t n, const TensorShape<N-1>& slice); //same syntax as for vector
	TensorShape(const TensorShape<N>& other) = default;
	TensorShape& operator=(const TensorShape& other) = delete;
    //assignment deleted because of const member variable
	~TensorShape() = default;
    const TensorShape<N-1>& operator[](const int i) const; //const because everything is const
    size_t size(void) const;
    size_t entryCount(void) const;
    std::vector<std::vector<size_t>> generateCoordinates(void) const;
	std::vector<std::vector<std::vector<size_t>>> generateSlicedCoordinates(void) const;
    const std::vector<TensorShape<N-1>> shape;
};


template<>
//a 1 tensor is a vector so this is just a const int
struct TensorShape<1>{
    TensorShape(const unsigned int size=0):shape(size){};
	TensorShape(const TensorShape<1>& other) = default;
	TensorShape& operator=(const TensorShape<1>& other) = delete;
	~TensorShape() = default;
    //assignment deleted because of const member variable
    size_t size(void)const {return shape;};
    size_t entryCount(void) const {return shape;};
    std::vector<std::vector<size_t>> generateCoordinates(void) const {
        std::vector<std::vector<size_t>> coordinates;
        for(size_t i=0; i<this->size(); i++){
            coordinates.push_back(std::vector<size_t>(1,i));
        }
        return coordinates;
    };
	std::vector<std::vector<std::vector<size_t>>> generateSlicedCoordinates(void) const{
		std::vector<std::vector<std::vector<size_t>>> coordinates(this->size());
		/*
        for(size_t i=0; i<this->size(); i++){	
            coordinates.push_back(std::vector<std::vector<size_t>>(1,std::vector<size_t>(1,i)));
        }
		*/
        return coordinates;	
	};
    const unsigned int shape;
};


//declaration and implementation of concatShapes
//are in tensor_template.hpp


//declaration and immplementation of comparison operators 
//operator==(const TensorShape<M> lhs&, const TensorShape<N> rhs&)
//operator!=(const TensorShape<M> lhs&, const TensorShape<N> rhs&)
//are in tensor_template.hpp



template<typename T, size_t N>
class Tensor{
public:
    Tensor(const std::vector<Tensor<T,N-1>>& entries);
    Tensor(const TensorShape<N>& shape);
	Tensor(const Tensor<T,N>& other) = default;
    ///assignment needs minimal attention because of const member variable
    //assignment throws if shapes don't match, otherwise default
    Tensor<T,N>& operator=(const Tensor<T,N>& other);
	~Tensor() = default;

    size_t size(void) const;
    Tensor<T,N-1>& operator[](const int i);
    const Tensor<T,N-1>& at(const int i) const;
    //const TensorShape<N>& shape(void) const not necessary anymore

    T& getEntry(const std::vector<size_t>& coordinates, const size_t depth=0); //returns entry at coordinates[depth:]

	
    const TensorShape<N> shape; //can be public because it is const 
private:
   
    std::vector<Tensor<T,N-1>> entries_;
};


template<typename T>
class Tensor<T,1>{
public:
    //default default constructor is ok
    Tensor(const std::vector<T>& entries);
    Tensor(int shape);
    Tensor(TensorShape<1> shape);
	Tensor(const Tensor<T,1>& other)=default;
    //assignment needs minimal attention because of const member variable
    //assignment throws if shapes don't match, otherwise default
    Tensor<T,1>& operator=(const Tensor<T,1>& other);
	~Tensor() = default;
    T& getEntry(const std::vector<size_t>& coordinates, const size_t depth=0);

    size_t size(void) const;
    T& operator[](const int i);
    const T& at(const int i) const;
    //const TensorShape<1>& shape(void); no longer required

    const TensorShape<1> shape; //just an int

private:
   
    std::vector<T> entries_;
};


//helper function to get shape from vector of N-1 tensors
template<typename T, size_t N>
TensorShape<N> getShape(const std::vector<Tensor<T,N-1>>& entries);

template<typename T>
TensorShape<1> getShape(const Tensor<T,1>& tensor);



#include "tensor_template.hpp"
#endif /*TENSROR_TEMPLATE_H*/