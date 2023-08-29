//tensor_template.tpp

/*
contains implementation of tensor template
*/

#include <vector>
#include <stdexcept>



/*
//
//TENSORSHAPE<N>
//IMPLEMENTATION
//
*/


template<size_t N>
TensorShape<N>::TensorShape(const std::vector<TensorShape<N-1>>& shape)
    :shape_(shape)
    {}

template<size_t N>
const TensorShape<N-1>& TensorShape<N>::operator[](const int i) const {
    if(i<0 || i>shape_.size()-1){throw std::out_of_range("TensorShape<N>::operator[]");}
    return shape_[i];
}

template<size_t N>
size_t TensorShape<N>::size(void) const
    {
        return shape_.size();
    }


/*
//
//TENSORSHAPE<N>
//END OF IMPLEMENTATION
//
*/


/*
//
//OPERATORS == AND != FOR TENSORSHAPE
//
*/


template<size_t N>
inline bool operator==(const TensorShape<N>& lhs , const TensorShape<N>& rhs){
    if(lhs.size()!=rhs.size()){return false;}
    for(size_t i=0; i<lhs.size();i++){
        if(lhs[i]!=rhs[i]){return false;}
    }
    return true;
}

inline bool operator==(const TensorShape<1>& lhs, const TensorShape<1>& rhs){
    return lhs.shape_==rhs.shape_;
}

template<size_t M, size_t N>
inline bool operator==(const TensorShape<M>& lhs, const TensorShape<N>& rhs){
    return false;
}

template<size_t N>
inline bool operator!=(const TensorShape<N>& lhs , const TensorShape<N>& rhs){
    return !(lhs==rhs);
}

inline bool operator!=(const TensorShape<1>& lhs, const TensorShape<1>& rhs){
    return !(lhs.shape_==rhs.shape_);
}


template<size_t M, size_t N>
inline bool operator!=(const TensorShape<M>& lhs, const TensorShape<N>& rhs){
    return true;
}

/*
//
//END OF
//OPERATORS == AND != FOR TENSORSHAPE
//
*/


/*
//
//TENSOR<T,N>
//IMPLEMENTATION
//
*/

//constructor
template<typename T, size_t N>
Tensor<T,N>::Tensor(const std::vector<Tensor<T,N-1>>& entries)
    :entries_(entries),shape_(getShape<T,N>(entries))
    {}


//copy assignment
template<typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator=(const Tensor<T,N>& other){
    if(shape_!=other.shape_){throw std::invalid_argument("Tensor<T,N> operator=:shapes do not match");}
    entries_=other.entries_;
    return *this;
}

//size
template<typename T, size_t N>
size_t Tensor<T,N>::size(void) const{
    return entries_.size();
}



//element access
template<typename T, size_t N>
Tensor<T,N-1>& Tensor<T,N>::operator[](const int i){
    if(i<0 || i>entries_.size()){
        throw std::out_of_range("Tensor<T,N>::operator[]");
    }
    return entries_[i];
}

//const element access
template<typename T, size_t N>
const Tensor<T,N-1>& Tensor<T,N>::at(const int i) const{
    if(i<0 || i>entries_.size()-1){throw std::out_of_range("Tensor<T,N>::at");}
    return entries_.at(i);
}

//shape access
template<typename T, size_t N>
const TensorShape<N>& Tensor<T,N>::shape(void) const{
    return shape_;
}



/*
//
//TENSOR<T,N>
//END OF IMPLEMENTATION
//
*/





/*
//
//TENSOR<T,1>
//IMPLEMENTATION
//
*/


//constructors
template<typename T>
Tensor<T,1>::Tensor(const std::vector<T>& entries)
    :entries_(entries), shape_(entries.size())
    {}

template<typename T>
Tensor<T,1>::Tensor(int shape)
    :entries_(shape), shape_(shape)
    {}

template<typename T>
Tensor<T,1>::Tensor(TensorShape<1> shape)
    :entries_(shape),shape_(shape)
    {}

template<typename T>
Tensor<T,1>& Tensor<T,1>::operator=(const Tensor<T,1>& other){
    if(shape_!=other.shape_){throw std::invalid_argument("Tensor<T,1> operator=:shapes do not match");}
    entries_=other.entries_;
    return *this;
}

//element access
template<typename T>
T& Tensor<T,1>::operator[](const int i){
    if(i<0 || i>shape_.shape_-1){
        throw std::out_of_range("Tensor<T,1>::operator[]");
    }
    return entries_[i];
}

//const element access
template<typename T>
const T& Tensor<T,1>::at(const int i) const{
    if(i<0 || i>entries_.size()-1){throw std::out_of_range("Tensor<T,1>::at");}
    return entries_.at(i);
}


template<typename T>
size_t Tensor<T,1>::size(void) const{
    return entries_.size();
}

template<typename T>
const TensorShape<1>& Tensor<T,1>::shape(void){
    return shape_;
}





/*
//
//TENSOR<T,1>
//END OF IMPLEMENTATION
//
*/


/*
//
//Helper functions to get shape from entries in constructor
//
*/

template<typename T, size_t N>
TensorShape<N> getShape(const std::vector<Tensor<T,N-1>>& entries){
    std::vector<TensorShape<N-1>> shapeVec;
    for(size_t i=0; i<entries.size(); i++){
        shapeVec.push_back(getShape(entries.at(i)));
    }
    TensorShape<N> shape(shapeVec);
    return shape;
}

template<typename T>
TensorShape<1> getShape(const Tensor<T,1>& tensor){
    TensorShape<1> shape(tensor.size());
    return shape;
}