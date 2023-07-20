//backpropagation.cpp

#include <vector>
#include "graphNeural.h"
#include "helper_fcts.h"

void GraphNN2DImage::updateState(
    const tensor_2d& sample,
    tensor_3d& currState,
    tensor_1d& outputLayerState
)
{
    //update graphical layers
    currState[0]=sample;
    for(size_t d=1; d<currState.size(); d++){
        for(size_t i=0; i<vertSize_; i++){
            for(size_t j=0; j<horSize_; j++){
                currState[d][i][j]=
                    weights_a_[d-1][i][j]*currState[d-1][i][j]
                    +weights_b_[d-1][i][j]*neighborContribution(i,j,currState[d-1])
                    +weights_c_[d-1][i][j]
                ;
            }
        }
    }
    
    //update outputLayer
    for(size_t l = 0; l<outputClassesCount_; l++){
        outputLayerState[l]=weights_output_bias_[l];
        for(size_t i = 0; i<vertSize_; i++){
            for(size_t j=0; j<horSize_; j++){
                outputLayerState[l]+=weights_output_[i][j][l]*sigma(currState[depth_][i][j]);
            }
        }
    }

}


void GraphNN2DImage::neuronGradient(
    const tensor_3d& currState, //used to store value of input to neurons at sample
    const tensor_1d& outputLayerState, //used to store value in output layer before taking softmax
    const tensor_2d& sample,
    const int sampleOutput,
    tensor_3d& neuronGradient //shape is (depth_+1),vertSize_,horSize_, layer 0 is input layer
)
{

    
   

    for(size_t i = 0; i<vertSize_; i++){
        for(size_t j=0; j<horSize_; j++){
            tensor_1d grad = softArgMaxGradComponent(outputLayerState, sampleOutput);
            double inv = 0;
            for(size_t l=0; l<outputClassesCount_; l++){
                double cont = grad[l]*weights_output_[i][j][l]*sigmaPrime(currState[depth_][i][j]);
                inv+=cont;
            }
            neuronGradient[depth_][i][j]=-1/inv;
        }
    }

    //compute remaining layers
    for(int k=depth_-1; k>-1; k--){
        //compute neuronGadient at depth k
        for(int i=0; i<vertSize_; i++){
            for(int j=0; j<horSize_; j++){
                //compute neuronGradient[k][i][j]
                std::vector<std::pair<int,int>> nbhd = getNeighbors(vertSize_, horSize_, i, j);
                neuronGradient[k][i][j]=neuronGradient[k+1][i][j]*weights_a_[k][i][j]*sigmaPrime(currState[k][i][j]); //contribution from same pixel
                for(const std::pair<int,int> nb : nbhd){
                    //contribution through neighbors
                    std::vector<std::pair<int,int>> nb_nbhd =  getNeighbors(vertSize_, horSize_, nb.first, nb.second);
                    auto it = std::find(nb_nbhd.begin(),nb_nbhd.end(),std::pair<int,int>{i,j}); 
                    int idx = it - nb_nbhd.begin(); //index at which [i,j] sits in neighbors of nb
                    std::vector<double> activatedNbs;
                    for(const auto nb_nb : nb_nbhd){
                        activatedNbs.push_back(sigma(currState[k][nb_nb.first][nb_nb.second]));
                    }
                    std::vector<double> softMaxGrad = BoltzmannOperatorGrad(activatedNbs , alpha_);
                    neuronGradient[k][i][j]+=
                        neuronGradient[k+1][nb.first][nb.second]
                        *weights_b_[k][nb.first][nb.second]
                        *sigmaPrime(currState[k][i][j])
                        *softMaxGrad[idx];
                }
            }
        }
    
    }
    return;
}




/// @brief computes gradient of weights at currDepth at samples
/// @param weights (inhomogeneous) tensor of weights
/// @param samples samples used for computation
/// @param weightsGradients stored gradients of weights, up to date at sample at depths>currDepth
/// @param currDepth current depth
void weightsLayerGradient(
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<std::vector<double>>>& samples,
    std::vector<std::vector<std::vector<double>>>& weightsGradients,
    int currDepth
){
    int depth=weightsGradients.size();
    if(currDepth==depth){
        //
    }

    //finish, easy based on neuron Gradient above


    return;

}


/// @brief computes gradient of weights at given samples
/// @param weights (inhomogeneous) tensor of weights
/// @param samples samples used for computation
/// @param weightsGradients stored gradients of weights updated for current position
void weightsGradient(
    const std::vector<std::vector<double>>& weights,
    const std::vector<std::vector<std::vector<double>>>& samples,
    std::vector<std::vector<std::vector<double>>>& weightsGradients
    ){
    int depth = weightsGradients.size()-1;
    for(int i=depth; i>-1; i--){
        weightsLayerGradient(weights, samples, weightsGradients, i);
    }
    return;
}




