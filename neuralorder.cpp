#include "neuralorder.h"

NeuralOrder::NeuralOrder(int inputs, int hidden)
{
    vector<int> row_on_count;
    vector<int> row_Off_count;
    vector<int> state_off;
    vector<float> integrator_init;
    for(int i = 0; i < inputs+hidden; i++){
        vector<vector<int>> neuroMatrix_On;
        vector<vector<int>> neuroMatrix_Off;
        for(int k = 0; k < (inputs+hidden)*2; k++){
            vector<int> row_on;
            vector<int> row_Off;
            for(int l = 0; l < (inputs+hidden)*2; l++){
                row_on.push_back(0.0);
                row_Off.push_back(0.0);
            }
            neuroMatrix_On.push_back(row_on);
            neuroMatrix_Off.push_back(row_Off);
        }
        orderMatrixOn.push_back(neuroMatrix_On);
        orderMatrixOff.push_back(neuroMatrix_Off);

        row_on_count.push_back(1);
        row_Off_count.push_back(1);
        state_off.push_back(0);
        integrator_init.push_back(0.0);
    }
    num_samples_On = row_on_count;
    num_samples_Off = row_Off_count;

    state = state_off;
    lastState = state_off;
    beforeLastState = state_off;

    integrator = integrator_init;
}

vector<int> NeuralOrder::propergate(vector<int> input){

    for(int i = 0; i < input.size(); i++){
        state[i] = input[i];
    }
    beforeLastState = lastState;
    lastState = state;

    for(int i = 0; i < lastState.size(); i++){
        float score_on = 0;
        float score_off = 0;
        for(int k = 0; k < (lastState.size()); k++){
            for(int l = 0; l < (lastState.size()); l++){
                if(k != l && (lastState[k] != beforeLastState[k] || lastState[l] != beforeLastState[l])){
                score_on += (1.0*orderMatrixOn[i][k*2+lastState[k]][l*2+lastState[l]])/(1.0*num_samples_On[i]);
                score_off += (1.0*orderMatrixOff[i][k*2+lastState[k]][l*2+lastState[l]])/(1.0*num_samples_Off[i]);
                }
            }
        }
        //integrator[i] *= 0.9;
        integrator[i] += (score_on-score_off);
        if( 1.0/(1.0+exp(-((integrator[i])/(0.01)))) > 1.0*rand()/RAND_MAX) state[i] = 1;
        else state[i] = 0;
    }


    for(int i = 0; i < input.size(); i++){
        state[i] = input[i];
    }

    return state;

}

void NeuralOrder::resetStates(){
    for(int i = 0; i < lastState.size(); i++){
        beforeLastState[i] = rand()%2;
        lastState[i] = rand()%2;
        state[i] = rand()%2;
        integrator[i] = 0;
    }
}

void NeuralOrder::train(vector<int> input){

    for(int i = 0; i < input.size(); i++){
        state[i] = input[i];
    }


    for(int i = 0; i < state.size(); i++){
        if(state[i] == 1) num_samples_On[i]++;
        else num_samples_Off[i]++;
        for(int k = 0; k < (state.size()); k++){
            for(int l = 0; l < (state.size()); l++){
                if(state[i] == 1 && k != l && (lastState[k] != state[k] || lastState[l] != state[l]) ){
                    orderMatrixOn[i][k*2+lastState[k]][l*2+lastState[l]]++;
                }
                if(state[i] == 0 && k != l && (lastState[k] != state[k] || lastState[l] != state[l])){
                    orderMatrixOff[i][k*2+lastState[k]][l*2+lastState[l]]++;
                }
            }
        }
    }

}

