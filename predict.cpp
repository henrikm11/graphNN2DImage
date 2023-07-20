//predict.cpp

#include <algorithm>
#include "graphNeural.h"
#include "helper_fcts.h"


double GraphNN2DImage::neighborContribution(const int vertPos, const int horPos, const tensor_2d& state){
    std::vector<std::pair<int,int>> neighbors = getNeighbors(vertSize_, horSize_, vertPos, horPos);
    std::vector<double> neighborStates;
    for(const auto nb : neighbors){
        neighborStates.push_back(state[nb.first][nb.second]);
    }
    return BoltzmannOperator(neighborStates);
}

//forward propagation
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
    
    return;
}

int GraphNN2DImage::predict(const tensor_2d& input){
    std::vector<double> outputProbs = predictProb(input);
    auto it = std::max_element(outputProbs.begin(), outputProbs.end());
    return it-outputProbs.begin();
}

std::vector<double> GraphNN2DImage::predictProb(const tensor_2d& input){
    
    tensor_3d graphStates(depth_+1, input); //states including input at graphical layers
    tensor_1d outputLayerState(outputClassesCount_,0);
    updateState(input, graphStates, outputLayerState);
    return softArgMax(outputLayerState);
}


std::vector<int> GraphNN2DImage::predict(const tensor_3d& inputs){
    std::vector<int> res(inputs.size(),0);
    size_t pos=0;
    for(auto input : inputs){
        res[pos]=predict(input);
        pos++;
    }
    return res;
}