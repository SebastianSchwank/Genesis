#include "neuralorder.h"

NeuralOrder::NeuralOrder(int inputs, int hidden)
{
    vector<int> row_count;
    vector<int> row_Off_count;
    vector<int> state_off;
    vector<float> integrator_init;
    for(int i = 0; i < inputs+hidden; i++){
        vector<vector<float>> neuroMatrix;
        for(int k = 0; k < (inputs+hidden)*2; k++){
            vector<float> row;
            for(int l = 0; l < (inputs+hidden)*2; l++){
                row.push_back(0);
            }
            neuroMatrix.push_back(row);
        }
        interpretationMatrix.push_back(neuroMatrix);

        row_count.push_back(1);
        row_Off_count.push_back(1);
        state_off.push_back(0);
        integrator_init.push_back(0.0);
    }
    num_samples_on = row_count;
    num_samples_off = row_count;

    state = state_off;
    lastState = state_off;
    beforeLastState = state_off;

    threshhold = integrator_init;

    integrator = integrator_init;
}

vector<int> NeuralOrder::propergate(vector<int> input){

    for(int i = 0; i < input.size(); i++){
        state[i] = input[i];
    }
    state[state.size()-1] = 1-state[state.size()-1];

    beforeLastState = lastState;
    lastState = state;

    for(int i = 0; i < lastState.size(); i++){
        float score_on = 0;
        float score_off = 0;
        for(int k = 0; k < (lastState.size()); k++){
            for(int l = 0; l < (lastState.size()); l++){
                if(k != l && (lastState[k] != beforeLastState[k] || lastState[l] != beforeLastState[l]) ){
                integrator[i] *= 0.99999;
                score_on += 1.0*interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]]/(1.0*num_samples_on[i]);
                score_on += 1.0*interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]]/(1.0*num_samples_off[i]); //+(1.0*orderMatrixOn[i][k*2+lastState[k]][l*2+lastState[l]]-1.0*orderMatrixOn[i][k*2+(1+lastState[k])%2][l*2+(1+lastState[l])%2])*1.0/(1.0*num_samples_On[i]);
                }
            }
        }
        integrator[i] += (score_on);
        if( 1.0/(1.0+exp(-((integrator[i])/(0.001)))) > 1.0*rand()/RAND_MAX) state[i] = 1;
        else state[i] = 0;
    }


    for(int i = 0; i < input.size(); i++){
        state[i] = input[i];
    }

    return state;

}

void NeuralOrder::resetStates(){
    for(int i = 0; i < lastState.size(); i++){
        beforeLastState[i] = 0;rand()%2;
        lastState[i] = 0;rand()%2;
        state[i] = 0;rand()%2;
        integrator[i] = 0;
    }
}

void NeuralOrder::train(){

    for(int i = 0; i < state.size(); i++){
        if(state[i] == 1) num_samples_on[i]++;
        else num_samples_off[i]++;
        for(int k = 0; k < (state.size()); k++){
            for(int l = 0; l < (state.size()); l++){
                if(state[i] == 1 && k != l && (lastState[k] != beforeLastState[k] || lastState[l] != beforeLastState[l]) ){
                    interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]] = (1.0*num_samples_on[i]*interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]]+1.0)/(1.0*num_samples_on[i]+1.0);
                }
                if(state[i] == 0 && k != l && (lastState[k] != beforeLastState[k] || lastState[l] != beforeLastState[l])){
                    interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]] = (1.0*num_samples_off[i]*interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]]-1.0)/(1.0*num_samples_off[i]+1.0);

                }
            }
        }
    }

}

