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

    GraphNN2DImage(int depth, double learningRate, int outputClassesCount, int horSize, int vertSize)
        :depth_(depth),
        learningRate_(learningRate),
        outputClassesCount_(outputClassesCount),
        horSize_(horSize),
        vertSize_(vertSize)
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

    double neighborContribution(const int vertPos, const int horPos, const tensor_2d& state);



    const unsigned int horSize_;
    const unsigned int vertSize_;
    const unsigned int outputClassesCount_;
    const unsigned int depth_; //number of graphical layers
    const double learningRate_;

    tensor_3d weights_a_; //weights connecting pixel to itself from layer to next
    tensor_3d weights_b_; //pixel to neighbors
    tensor_3d weights_c_; //intercept
    tensor_3d weights_output_; //weights_output[i][j][o] from pixel i j to class o
    tensor_1d weights_output_bias_;
};


#endif // GRAPHNEURAL_H 