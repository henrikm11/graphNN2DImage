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

int GraphNN2DImage::predict(const tensor_2d& input){
    std::vector<double> outputProbs = predictProb(input);
    auto it = std::max_element(outputProbs.begin(), outputProbs.end());
    return it-outputProbs.begin();
}

std::vector<double> GraphNN2DImage::predictProb(const tensor_2d& input){
    
    tensor_3d states(depth_+1, input); //states including input at graphical layers

    for(size_t d=1; d<depth_+1; d++){
        for(size_t i=0; i<vertSize_; i++){
            for(size_t j=0; j<horSize_; j++){
                states[d][i][j]=
                    weights_a_[d-1][i][j]*states[d-1][i][j]
                    +weights_b_[d-1][i][j]*neighborContribution(i,j,states[d-1])
                    +weights_c_[d-1][i][j]
                ;
            }
        }
    }

    std::vector<double> outputPreNorm(outputClassesCount_,0); //states at output layer before normalization
    for(size_t l = 0; l<outputClassesCount_; l++){
        for(size_t i=0; i<vertSize_; i++){
            for(size_t j=0; j<horSize_; j++){
                outputPreNorm[l]+=weights_output_[i][j][l]*states[depth_][i][j]+weights_output_bias_[l];   
            }
        }
    }

    return softArgMax(outputPreNorm);
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