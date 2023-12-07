//tensor_template.tpp

/*
contains implementation of tensor template
*/


/*
// TODO
// -) implement slice operator of type Tensor<T,M>& getSlice(const vector<size_t>&)?
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
    :shape(shape)
    {}

template<size_t N>
TensorShape<N>::TensorShape(size_t n, const TensorShape<N-1>& slice)
    :shape(n, slice)
    {}


template<size_t N>
const TensorShape<N-1>& TensorShape<N>::operator[](const int i) const {
    if(i<0 || i>shape.size()-1){throw std::out_of_range("TensorShape<N>::operator[]");}
    return shape[i];
}

template<size_t N>
size_t TensorShape<N>::size(void) const
    {
        return shape.size();
    }

template<size_t N>
size_t TensorShape<N>::entryCount() const {
    size_t count=0;
    for(size_t i=0; i<shape.size(); i++){
        count+=shape[i].entryCount();
    }
    return count;
}
	

template<size_t N>
std::vector<std::vector<size_t>> TensorShape<N>::generateCoordinates() const{
    std::vector<std::vector<size_t>> coordinates;
    for(size_t i=0; i<shape.size();i++){
        std::vector<std::vector<size_t>> tails = shape[i].generateCoordinates();
        for(auto& tail : tails){
            std::vector<size_t> coord = {i};
            coord.insert(coord.end(),tail.begin(),tail.end());
            coordinates.push_back(coord);
        }
    }
    return coordinates;
}

template<size_t N>
std::vector<std::vector<std::vector<size_t>>> TensorShape<N>::generateSlicedCoordinates() const{
std::vector<std::vector<std::vector<size_t>>> coordinates;
    for(size_t i=0; i<shape.size();i++){
        std::vector<std::vector<size_t>> coordinatesSlice;
        std::vector<std::vector<size_t>> tails = shape[i].generateCoordinates();
        for(auto& tail : tails){
            std::vector<size_t> coord;// = {i};
            coord.insert(coord.end(),tail.begin(),tail.end());
            coordinatesSlice.push_back(coord);
        }
        coordinates.push_back(coordinatesSlice);
    }
    return coordinates;  
}






/*
//
//TENSORSHAPE<N>
//END OF IMPLEMENTATION
//
*/


/*
//
//CONCATESHAPES FUNCTIONS
//
*/

template<size_t M, size_t N>
inline TensorShape<M+N> concatShapes(const TensorShape<M>& first, const TensorShape<N>& second){
    std::vector<TensorShape<M+N-1>> concatEntries;
    for(size_t i=0; i<first.size();i++){
        concatEntries.emplace_back(concatShapes(first[i],second));
    }
    TensorShape<M+N> concatShape(concatEntries);
    return concatShape;
}


template<size_t N>
inline TensorShape<N+1> concatShapes(const TensorShape<1>& first, const TensorShape<N>& second){
    std::vector<TensorShape<N>> concatEntries(first.size(),second);
    TensorShape<N+1> concatShape(concatEntries);
    return concatShape;
}

/*
//
//END OF CONCATESHAPES FUNCTIONS
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
    return lhs.shape==rhs.shape;
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
    return !(lhs.shape==rhs.shape);
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
    :entries_(entries),shape(getShape<T,N>(entries))
    {}

template<typename T, size_t N>
Tensor<T,N>::Tensor(const TensorShape<N>& shape)
    :shape(shape)
    {
        //entries default initialized to length 0, now fill it
        for(size_t i=0; i<shape.size(); i++){
            Tensor<T,N-1> entry(shape[i]);
            entries_.push_back(entry);
        }
    }


//copy assignment
template<typename T, size_t N>
Tensor<T,N>& Tensor<T,N>::operator=(const Tensor<T,N>& other){
    if(this==&other){return *this;}
    if(shape!=other.shape){throw std::invalid_argument("Tensor<T,N> operator=:shapes do not match");}
    entries_=other.entries_;
    return *this;
}

//size
template<typename T, size_t N>
size_t Tensor<T,N>::size(void) const{
    return entries_.size();
}




//slice access
template<typename T, size_t N>
Tensor<T,N-1>& Tensor<T,N>::operator[](const int i){
    if(i<0 || i>entries_.size()){
        throw std::out_of_range("Tensor<T,N>::operator[]");
    }
    return entries_[i];
}

//const slice access
template<typename T, size_t N>
const Tensor<T,N-1>& Tensor<T,N>::at(const int i) const{
    if(i<0 || i>entries_.size()-1){throw std::out_of_range("Tensor<T,N>::at");}
    return entries_.at(i);
}

/*
//shape access
template<typename T, size_t N>
const TensorShape<N>& Tensor<T,N>::shape(void) const{
    return shape;
}
*/


//element access
///@brief returns reference to entry at coordinates[depth:]
template<typename T, size_t N>
T& Tensor<T,N>::getEntry(const std::vector<size_t>& coordinates, const size_t depth){
    if(coordinates.size()-depth!=N){
        throw(std::domain_error("Tensor<T,N>::getEntry: size of coordinates, depth, N do not mathc"));
    }
    if(coordinates[depth]>shape.size()){
        throw(std::out_of_range("Tensor<T,N>::getEntry"));
    }
    return entries_[coordinates[depth]].getEntry(coordinates,depth+1);
};


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
    :entries_(entries), shape(entries.size())
    {}

template<typename T>
Tensor<T,1>::Tensor(int shape)
    :entries_(shape), shape(shape)
    {}

template<typename T>
Tensor<T,1>::Tensor(TensorShape<1> shape)
    :entries_(shape.size()),shape(shape)
    {}

//copy assignment
template<typename T>
Tensor<T,1>& Tensor<T,1>::operator=(const Tensor<T,1>& other){
    if(this==&other){return *this;}
    if(shape!=other.shape){throw std::invalid_argument("Tensor<T,1> operator=:shapes do not match");}
    entries_=other.entries_;
    return *this;
}

//element access
template<typename T>
T& Tensor<T,1>::operator[](const int i){
    if(i<0 || i>shape.shape-1){
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

/*
template<typename T>
const TensorShape<1>& Tensor<T,1>::shape(void){
    return shape;
}
*/


//element access
///@brief returns reference to entry at coordinates[depth:]
template<typename T>
T& Tensor<T,1>::getEntry(const std::vector<size_t>& coordinates, const size_t depth){
    if(coordinates.size()-depth!=1){
        throw(std::domain_error("Tensor<T,N>::getEntry: size of coordinates, depth, N do not mathc"));
    }
        if(coordinates[depth]>shape.size()){
        throw(std::out_of_range("Tensor<T,N>::getEntry"));
    }
    return entries_[coordinates[depth]];
};


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