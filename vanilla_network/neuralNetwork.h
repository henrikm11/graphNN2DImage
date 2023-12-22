//neuralNetwork.h
//template for basic neural network


/*
TO DO
-) neuralNetwork is all public right now for testing/debugging
-) throw if network consits only of input layer
-) add gradient of costFct_
-) note to self: keep costFct_ on tensor level: mse can be implemented on entry leve, cross log entropy for instance cant
*/

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "tensor_template/tensor_template.h"
#include "helper_fcts.h"


typedef std::vector<std::vector<std::vector<size_t>>> slicedCoordTensor; 


/*
//conventions:

//weights w_{i,c,d} 
//is from layer i to i+1 from neuron at c to neuron at d

//bias b_{d}
// is bias for feeding into neuron at d

*/


//template parameters describe depth of tensors for input, output and hidden layers
//in and output tensors present as slices of network tensor which is of size N
template<size_t N>
class neuralNetwork{
public:
    neuralNetwork(
        const TensorShape<N>& shape,
        double (*const activationFunction)(const double),
        double (*const activationFunctionGrad)(const double),
        Tensor<double,N-1> (*const outputLayerTransform_)(const Tensor<double,N-1>&),
        Tensor<double,2*N-2>(*const outputLayerTransformGrad_)(const Tensor<double,N-1>&),
        double (*const errorFct)(const Tensor<double,N-1>&, const Tensor<double,N-1>&),
        double (*const errorFctGrad_)(const Tensor<double,N-1>&, const Tensor<double,N-1>&, const std::vector<size_t>&, size_t)
        );

    neuralNetwork(
        const TensorShape<N>& shape,
        double (*const activationFunction)(const double),
        double (*const activationFunctionGrad)(const double),
        double (*const errorFct)(const Tensor<double,N-1>&, const Tensor<double,N-1>&),
        double (*const errorFctGrad_)(const Tensor<double,N-1>&, const Tensor<double,N-1>&, const std::vector<size_t>&, size_t)
        );

    //Tensor<double, N-1> predict(const Tensor<double,N-1>& input); 
    //void setLearningRate(const double rate);
    /*
    void fit(
        const std::vector<Tensor<double,N-1>>& sampleInputs,
        const std::vector<Tensor<double,N-1>>& sampleOutputs
    );
    */
	
    Tensor<double,N-1> predict(const Tensor<double,N-1>& input, bool updateNeuronStates=true);
    
    void fit(
        const std::vector<Tensor<double,N-1>>& trainingIns, 
        const std::vector<Tensor<double,N-1>>& trainingOuts,
        double learningRate=0.1,
        size_t maxEpochs=100,
        size_t batchSize=64    
    );
	
    //make those private again after testing
//private only gone for the moment for testing/debugging purposes
//private:

    //variables related to basic architecture of the network
	const size_t neuronCount_;
    const size_t outputLayerCount_;
    const TensorShape<N> networkShape_; //somewhat redundant as it is stored in neuronStates?
    double (*const activationFct_)(const double);
    double (*const activationFctGrad_)(const double);
    Tensor<double,N-1> (*const outputLayerTransform_)(const Tensor<double,N-1>&);
    Tensor<double,2*N-2>(*const outputLayerTransformGrad_)(const Tensor<double,N-1>&);
    //Tensor<double,N-1> (*const outputLayerTransformGrad_)(const Tensor<double,N-1>&, const std::vector<size_t>&);
    double (*const costFct_)(const Tensor<double,N-1>&, const Tensor<double,N-1>&);
    double (*const costFctGrad_)(const Tensor<double,N-1>&, const Tensor<double,N-1>&, const std::vector<size_t>&, size_t); //last entry is for normalization
    Tensor<double,2*N-1> weights_;
    Tensor<double,N> biases_;
    Tensor<double,2*N-1> weightsGrad_; 
    Tensor<double,N> biasesGrad_;
    Tensor<double,N> neuronStates_;
    Tensor<double,N> neuronGrad_;
    

 

    //training data
    std::vector<Tensor<double,N-1>> trainingIns_;
    std::vector<Tensor<double,N-1>> trainingOuts_;
    
    //double activationFunction_(const double input){return std::max((double)0,input);};
    //double activationFunctionGrad_(const double input);
    const slicedCoordTensor coordinates_;
	
	
    void updateNeuronStates_(const Tensor<double,N-1>& input);
    void updateLayer_(const size_t i);
    void updateOutputLayer_(void);
	void updateGradient_(const Tensor<double, N-1>& sampleIn, const Tensor<double,N-1>& sampleOut, size_t batchIdx=0);
	void updateOutputLayerGrad_(const Tensor<double, N-1>& sampleIn, const Tensor<double,N-1>& sampleOut);
	void backpropagation_(void);
    void updateWeightsGrad_(size_t batchIdx=0);
    void updateBiasesGrad_(size_t batchIdx=0);
    void netGradientDescentStep_(double learningRate);


    void weightInitilization_(void);
    void biasInitilization_(void);

    void fit_(
        double learningRate=0.01,
        size_t maxEpochs=100,
        size_t batchSize=64   
    );
};





//helper function to generate TensorShape of weights for fully connected case
template<size_t N>
TensorShape<2*N-1> getWeightsShape(const TensorShape<N>& networkShape);






#include "neuralNetwork.hpp"
#include "gradientDescent.hpp"
#include "backpropagation.hpp"
#include "fit.hpp"
#include "weight_initilization.hpp"

#endif /* NEURAL_NETWORK_H */



