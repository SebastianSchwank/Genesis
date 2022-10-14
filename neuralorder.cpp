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
    last_num_samples_on = num_samples_on = row_count;
    last_num_samples_off = num_samples_off = row_Off_count;

    state = state_off;
    lastState = state_off;
    beforeLastState = state_off;

    integrator = integrator_init;
    stateChangeActivity = integrator_init;
    meanState = integrator_init;
}

vector<int> NeuralOrder::propergate(vector<int> input){

    for(int i = 0; i < input.size(); i++){
        state[i] = 1.0-input[i];
    }

    //state[state.size()-1] = 1-state[state.size()-1];

    beforeLastState = lastState;
    lastState = state;

    for(int i = 0; i < lastState.size(); i++){
        float score_on = 0;
        float score_off = 0;
        for(int k = 0; k < (lastState.size()); k++){
            for(int l = 0; l < (lastState.size()); l++){
                if( (lastState[k] != beforeLastState[k]) ){
                score_on += (1.0*interpretationMatrix[i][k*2+(lastState[k])%2][l*2+(lastState[l])%2])*stateChangeActivity[l]*stateChangeActivity[k]*state[k];
                }
            }
        }
        integrator[i] *= 0.9;
        integrator[i] += (score_on);
        if( 1.0/(1.0+exp(-((integrator[i])*(0.00001)))) >  1.0*rand()/RAND_MAX) state[i] = 1;
        else state[i] = 0;

        //stateChangeActivity[i] *= 0.99;
        stateChangeActivity[i] = 2.0/(1.0+exp(-stateChangeActivity[i]+(state[i]-lastState[i])))+1.0;

        meanState[i] = (meanState[i]*999.0+state[i])/1000.0;
    }


    vector<int> output;
    for(int i = 0; i < input.size(); i++){
        state[i] = 1.0-input[i];
    }

    for(int i = 0; i < state.size(); i++){
        output.push_back(1-signbit(integrator[i]));
    }

    return output;

}

void NeuralOrder::resetStates(){
    for(int i = 0; i < lastState.size(); i++){
        beforeLastState[i] = rand()%2;
        lastState[i] = rand()%2;
        state[i] = rand()%2;
        integrator[i] = 0;
        //meanState[i] = 0.5;
        //stateChangeActivity[i] = 0.0;
    }
}

void NeuralOrder::train(){

    for(int i = 0; i < state.size(); i++){

        for(int k = 0; k < (state.size()); k++){
            for(int l = 0; l < (state.size()); l++){
                //if(k != l && (lastState[k] != beforeLastState[k] ) ) interpretationMatrix[i][k*2+(lastState[k])%2][l*2+(lastState[l])%2] += (state[i]*meanState[i]-(1.0-state[i])*(1.0-meanState[i]))*(stateChangeActivity[l]*stateChangeActivity[k]);

                if((lastState[i] == 1)   && (lastState[k] != beforeLastState[k]) )
                    interpretationMatrix[i][k*2+(lastState[k])%2][l*2+(lastState[l])%2] += (1.0-abs(stateChangeActivity[i]))*(meanState[i])*(stateChangeActivity[k]+stateChangeActivity[l]);//*(1.0-2.0*signbit(integrator[l]))*(1.0-2.0*signbit(integrator[k]));
                if((lastState[i] == 0)  && (lastState[k] != beforeLastState[k]) )
                    interpretationMatrix[i][k*2+(lastState[k])%2][l*2+(lastState[l])%2] -= (1.0-abs(stateChangeActivity[i]))*(1.0-meanState[i])*(stateChangeActivity[k]+stateChangeActivity[l]);//*(1.0-2.0*signbit(integrator[l]))*(1.0-2.0*signbit(integrator[k]));

            }

        }


    }

}


void NeuralOrder::sleep(){
    vector<int> emptyV;

    this->propergate(emptyV);

    for(int i = 0; i < state.size(); i++){

        for(int k = 0; k < (state.size()); k++){
            for(int l = 0; l < (state.size()); l++){
                //if(k != l && (lastState[k] != beforeLastState[k] ) ) interpretationMatrix[i][k*2+(lastState[k])%2][l*2+(lastState[l])%2] += (state[i]*meanState[i]-(1.0-state[i])*(1.0-meanState[i]))*(stateChangeActivity[l]*stateChangeActivity[k]);

                if((lastState[i] == 1)   && (lastState[k] != beforeLastState[k]) )
                    interpretationMatrix[i][k*2+(lastState[k])%2][l*2+(lastState[l])%2] -= (meanState[i])*(stateChangeActivity[l])*state[k];//*(1.0-2.0*signbit(integrator[l]))*(1.0-2.0*signbit(integrator[k]));
                if((lastState[i] == 0)  && (lastState[k] != beforeLastState[k]) )
                    interpretationMatrix[i][k*2+(lastState[k])%2][l*2+(lastState[l])%2] += (1.0-meanState[i])*(stateChangeActivity[l])*state[k];//*(1.0-2.0*signbit(integrator[l]))*(1.0-2.0*signbit(integrator[k]));

            }

        }


    }

}
