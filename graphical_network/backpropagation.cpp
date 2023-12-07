//backpropagation.cpp

#include <vector>
#include "graphNeural.h"
#include "helper_fcts.h"




void GraphNN2DImage::neuronGradient(
    const tensor_2d& sample,
    const int sampleOutput,
    tensor_3d& neuronGrad, //shape is (depth_+1),vertSize_,horSize_, layer 0 is input layer
    const tensor_3d& currState, //used to store value of input to neurons at sample
    const tensor_1d& outputLayerState //used to store value in output layer before taking softmax
)
{
    //output layer explicitly
    for(size_t i = 0; i<vertSize_; i++){
        for(size_t j=0; j<horSize_; j++){
            tensor_1d grad = softArgMaxGradComponent(outputLayerState, sampleOutput);
            double inv = 0;
            for(size_t l=0; l<outputClassesCount_; l++){
                inv += grad[l]*weights_output_[i][j][l]*sigmaPrime(currState[depth_][i][j]);
            }
            neuronGrad[depth_][i][j]=-1/inv;
        }
    }

    //remaining layers, 0th layer is input layer
    for(int k=depth_-1; k>0; k--){
        //compute neuronGadient at depth k
        for(int i=0; i<vertSize_; i++){
            for(int j=0; j<horSize_; j++){
                //compute neuronGradient[k][i][j]
                std::vector<std::pair<int,int>> nbhd = getNeighbors(vertSize_, horSize_, i, j);
                neuronGrad[k][i][j]=neuronGrad[k+1][i][j]*weights_a_[k][i][j]*sigmaPrime(currState[k][i][j]); //contribution from same pixel
                for(const std::pair<int,int> nb : nbhd){
                    //contribution through neighbors
                    std::vector<std::pair<int,int>> nb_nbhd =  getNeighbors(vertSize_, horSize_, nb.first, nb.second);
                    auto it = std::find(nb_nbhd.begin(),nb_nbhd.end(),std::pair<int,int>{i,j}); 
                    int idx = it - nb_nbhd.begin(); //index at which [i,j] sits in neighbors of nb
                    std::vector<double> activatedNbs;
                    for(const auto nb_nb : nb_nbhd){
                        activatedNbs.push_back(sigma(currState[k][nb_nb.first][nb_nb.second]));
                    }
                    std::vector<double> softMaxGrad = BoltzmannOperatorGrad(activatedNbs , alpha_); //only need one component, can be optimized a bit
                    neuronGrad[k][i][j]+=
                        neuronGrad[k+1][nb.first][nb.second]
                        *weights_b_[k][nb.first][nb.second]
                        *sigmaPrime(currState[k][i][j])
                        *softMaxGrad[idx];
                }
            }
        }
    }
    return;
}


void GraphNN2DImage::weightsGradient(
    const tensor_2d& sample,
    const int sampleOutput,
    tensor_3d& weights_a_grad,
    tensor_3d& weights_b_grad,
    tensor_3d& weights_c_grad,
    tensor_3d& weights_output_grad,
    tensor_1d& weights_output_bias_grad,
    tensor_3d& neuronGrad,
    tensor_3d& currState, //used to store value of input to neurons at sample
    tensor_1d& outputLayerState //used to store value in output layer before taking softmax

){
    updateState(sample,  currState, outputLayerState); 
    neuronGradient(sample, sampleOutput, neuronGrad, currState, outputLayerState); 

    for(size_t d=0; d<depth_; d++){
        for(size_t i=0; i<vertSize_; i++){
            for(size_t j=0; j<horSize_; j++){
                weights_a_grad[d][i][j]=neuronGrad[d+1][i][j]*sigma(currState[d][i][j]);
                weights_b_grad[d][i][j]=neuronGrad[d+1][i][j]*neighborContribution(i,j,currState[d]); //neighborCont has sigma inbuilt
                weights_c_grad[d][i][j]=neuronGrad[d+1][i][j];
            }
        }
    }

    tensor_1d grad = softArgMaxGradComponent(outputLayerState, sampleOutput);
    for(size_t i=0; i<vertSize_; i++){
        for(size_t j=0; j<horSize_; j++){
            weights_output_grad[sampleOutput][i][j]=-1/(grad[sampleOutput]*sigma(currState[depth_][i][j]));
        }
    }
    weights_output_bias_grad[sampleOutput]=-1/grad[sampleOutput];
    return;
}
