#include "neuralorder.h"

NeuralOrder::NeuralOrder(int inputs, int hidden)
{
    vector<int> row_count;
    vector<int> row_Off_count;
    vector<int> state_off;
    vector<float> integrator_init;
    for(int i = 0; i < inputs+hidden; i++){
        vector<vector<float>> neuroMatrix;
        for(int k = 0; k < (inputs+hidden); k++){
            vector<float> row;
            for(int l = 0; l < (inputs+hidden); l++){
                row.push_back(0);
            }
            interpretationMatrix.push_back(row);
        }

        row_count.push_back(1);
        row_Off_count.push_back(1);
        state_off.push_back(0);
        integrator_init.push_back(0.0);
        Activity.push_back(0.0);
    }
    last_num_samples_on = num_samples_on = row_count;
    last_num_samples_off = num_samples_off = row_Off_count;

    state = state_off;
    lastState = state_off;
    beforeLastState = state_off;

    integrator = integrator_init;
    stateChangeActivity = integrator_init;
    meanState = integrator_init;
}

vector<float> NeuralOrder::propergate(vector<int> input){

    for(int i = 0; i < input.size(); i++){
        state[i] = (1.0-state[i])*input[i];
    }

    state[state.size()-1] = 1-state[state.size()-1];

    beforeLastState = lastState;
    lastState = state;

    for(int i = 0; i < lastState.size(); i++){
        float score_on = 0;
        float score_off = 0;
        for(int k = 0; k < (lastState.size()); k++){
                if( (lastState[k] != beforeLastState[k]) && i!=k ){
                score_on += (1.0*interpretationMatrix[i][k]);
                }
        }
        integrator[i] += (score_on);
        if( 1.0/(1.0+exp(-((integrator[i])*(1.0)))) >  1.0*rand()/RAND_MAX){ state[i] = 1-state[i]; integrator[i] *= -1.0; num_samples_off[i] = 0; if(lastState[i] == beforeLastState[i]) Activity[i] = abs(Activity[i])*-1; }
        else{ state[i] = 0; num_samples_off[i]++; }



        Activity[i] *= 0.99;
        Activity[i] += abs(state[i]-lastState[i]);

        //if(i == 16) cout << Activity[i] << "\n";

        meanState[i] = (meanState[i]*999.0+1.0/(1.0+exp(-Activity[i])))/1000.0;
    }


    vector<float> output;

    for(int i = 0; i < state.size(); i++){
        output.push_back(1.0/(1.0+exp(-Activity[i])));
    }

    return output;

}

void NeuralOrder::resetStates(){
    for(int i = 0; i < lastState.size(); i++){
        beforeLastState[i] = 0;rand()%2;
        lastState[i] = 0;rand()%2;
        state[i] = 0;rand()%2;
        integrator[i] = 0;
        Activity[i] = 0;
        //meanState[i] = 0.5;
        //stateChangeActivity[i] = 0.0;
    }
}

void NeuralOrder::train(){

    for(int i = 0; i < state.size(); i++){

        for(int k = 0; k < (state.size()); k++){

                if((lastState[i] != state[i])   && (lastState[k] != state[k] && i!=k)){
                    interpretationMatrix[k][i] += (1.0-1.0/(1.0+exp(-integrator[i]*integrator[k])))*(meanState[i]);
                }
                if((lastState[i] == state[i])  && (lastState[k] != state[k]) && i!=k){
                    interpretationMatrix[k][i] -= (1.0/(1.0+exp(-integrator[i]*integrator[k])))*(1.0-meanState[i]);
                }


        }


    }

}

vector<vector<float>>  NeuralOrder::getWeights(){
    return interpretationMatrix;
}

vector<float> NeuralOrder::getActivation(){
    return Activity;
}

void NeuralOrder::sleep(){
    vector<int> emptyV;

    this->propergate(emptyV);

    for(int i = 0; i < state.size(); i++){

        for(int k = 0; k < (state.size()); k++){

                if((lastState[i] != state[i])   && (lastState[k] != state[k] && i!=k)){
                    interpretationMatrix[k][i] -= (1.0-1.0/(1.0+exp(-Activity[i]*Activity[k])))*(meanState[i]);
                }
                if((lastState[i] == state[i])  && (lastState[k] != state[k]) && i!=k){
                    interpretationMatrix[k][i] += (1.0/(1.0+exp(-Activity[i]*Activity[k])))*(1.0-meanState[i]);
                }


        }


    }

}
