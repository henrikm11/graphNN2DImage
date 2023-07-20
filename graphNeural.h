//graphNeural.h

#ifndef GRAPHNEURAL_H
#define GRAPHNEURAL_H

#include <vector>
#include <cmath>
#include <vector>

typedef std::vector<double> tensor_1d;
typedef std::vector<std::vector<double>> tensor_2d;
typedef std::vector<std::vector<std::vector<double>>> tensor_3d;


class GraphNN2DImage{
public:

    GraphNN2DImage(
        unsigned int horSize, 
        unsigned int vertSize,
        unsigned int outputClassesCount, 
        unsigned int depth,
        double alpha=10,
        double learningRate=0.1
        )
        :horSize_(horSize),
        vertSize_(vertSize),
        outputClassesCount_(outputClassesCount),
        depth_(depth),
        alpha_(alpha),
        learningRate_(learningRate)  
    {};

    void fit();

    std::vector<int> predict(const tensor_3d& inputs);
    int predict(const tensor_2d& input);
    tensor_1d predictProb(const tensor_2d& input);


private:
    
    /// @brief 
    /// @param currState current value at neurons before activation
    /// @param samples samples used to update
    /// @param sampleOutputs outputs for those samples in same order
    /// @param neuronGradient gradient in direction of input to neurons (i.e before activation is used),neuronGradient[k][i][j] is derivative at depth k and pixel [i,j]
    void neuronGradient(
        const tensor_3d& currState,
        const tensor_2d& sample,
        const int sampleOutput,
        tensor_3d& neuronGradient
    );

    void updateState(
        tensor3d& currState,
        tensor1d& outputLayerState
    );

    double neighborContribution(const int vertPos, const int horPos, const tensor_2d& state);


    //data dependent parameters
    const unsigned int horSize_;
    const unsigned int vertSize_;
    const unsigned int outputClassesCount_;

    //model parameters
    const unsigned int depth_; 
    const double alpha_;    //regularization parameter for approximation of max, argmax
    const double learningRate_;

    tensor_3d weights_a_; //weights connecting pixel to itself from layer to next
    tensor_3d weights_b_; //pixel to neighbors
    tensor_3d weights_c_; //intercept
    tensor_3d weights_output_; //weights_output[i][j][o] from pixel i j to class o
    tensor_1d weights_output_bias_;
};


#endif // GRAPHNEURAL_H 