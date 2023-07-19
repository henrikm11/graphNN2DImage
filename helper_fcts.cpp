//helper_fcts.cpp

#include "graphNeural.h"

std::vector<std::pair<int,int>> getNeighbors(int vertSize, int horSize, int vertPos, int horPos){
    std::vector<std::pair<int,int>> neighbors;
    for(int i=-1; i<2; i++){
        for(int j=-1; j<2; j++){
            int newHorPos=horPos+i;
            int newVertPos=vertPos+j;

            if(i==0 && j==0){continue;} //same pos
            if(newHorPos<0){continue;} //too far left etc
            if(newHorPos>horPos){continue;}  
            if(newVertPos<0){continue;}
            if(newVertPos>vertSize){continue;}

            neighbors.push_back({newVertPos,newHorPos});
        }
    }
    return neighbors;
}
