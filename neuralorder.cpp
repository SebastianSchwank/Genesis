#include "neuralorder.h"

NeuralOrder::NeuralOrder(int inputs, int hidden)
{
    vector<int> row_count;
    vector<int> row_Off_count;
    vector<int> state_off;
    vector<float> integrator_init;
    for(int i = 0; i < inputs+hidden; i++){
        vector<vector<int>> neuroMatrix;
        for(int k = 0; k < (inputs+hidden)*2; k++){
            vector<int> row;
            for(int l = 0; l < (inputs+hidden)*2; l++){
                row.push_back(0);
            }
            neuroMatrix.push_back(row);
        }
        interpretationMatrix.push_back(neuroMatrix);

        row_count.push_back(1);
        row_Off_count.push_back(-1);
        state_off.push_back(0);
        integrator_init.push_back(0.0);
    }
    num_samples_on = row_count;
    num_samples_off = row_Off_count;

    state_balance = state_off;
    state = state_off;
    lastState = state_off;
    beforeLastState = state_off;


    integrator = integrator_init;

    numSamples = 0;
    globalState = 0;
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

                score_on += (1.0*interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]])*abs((1.0*interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]]-1.0*interpretationMatrix[i][k*2+(lastState[k]+1)%2][l*2+(lastState[l]+1)%2]));}
            }
        }
        integrator[i] *= 0.9;
        integrator[i] += (score_on);
        if( 1.0/(1.0+exp(-((integrator[i])*(0.001)))) >  1.0*rand()/RAND_MAX) state[i] = 1-state[i];
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

    numSamples++;
    globalState = 0;

    for(int i = 0; i < state.size(); i++){
        int absMin = INFINITY;
        if(state[i] == 1){num_samples_on[i]++; }
        else{ num_samples_off[i]++; }

        for(int k = 0; k < (state.size()); k++){
            for(int l = 0; l < (state.size()); l++){
                if(state[i] == 1 && k != l && (lastState[k] != beforeLastState[k] || lastState[l] != beforeLastState[l]) ){
                    interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]] += num_samples_off[i];
                }
                if(state[i] == 0 && k != l && (lastState[k] != beforeLastState[k] || lastState[l] != beforeLastState[l])){
                    interpretationMatrix[i][k*2+lastState[k]][l*2+lastState[l]] -= num_samples_on[i];
                }
            }
        }


    }

}
