#include "neuralcluster.h"


NeuralCluster::NeuralCluster(int inputs, int outputs, int hidden, int attention){

    numInputs = inputs;
    numOutputs = outputs;
    numHiddens = hidden;
    numRekurrent = attention;

    vector<vector<float> > weightsCreation;
    for(int i = 0; i < inputs+outputs+hidden+attention+1; i++){
        fireCounter.push_back(1.0);
        counterActivation.push_back(1.0);
        lastCounter.push_back(1.0);
        polarityCounter.push_back(1.0);
        fireReal.push_back(1.0);
        realActivation.push_back(1.0);
        lastReal.push_back(1.0);
        beforelastReal.push_back(1.0);
        polarityReal.push_back(1.0);
        period.push_back(rand()%maxPeriod+1);
        counter.push_back(rand()%period[i]+1);
        slowness.push_back(1.0);
        momentumVector.push_back(0.0);
        //if(i > inputs+outputs) slowness[i] = 1.0*rand()/RAND_MAX;
        //slowness[i] = slowness[i]*slowness[i]*slowness[i];
        vector<float> weightColumnActive;
        vector<float> weightColumnInactive;

        vector<float> momentumColumn;
        vector<int> firingColumn;
        vector<float> deltaColumn;
        vector<float> relativeBehaveColumn;
        vector<int> type;


        int neuronType = -1;//(rand()%(numOutputs+1))-1;
        vector<bool> maskV;
        for(int j = 0; j < inputs+outputs+hidden+attention+1; j++){
            //if((j+i)%2 ==0) weightColumn.push_back(-0.001);
            //else  weightColumn.push_back(0.001);

            //weightColumn.push_back(0.01);

            weightColumnActive.push_back(2.0*((1.0*rand()/RAND_MAX-0.5)));
            weightColumnInactive.push_back(0.2*((1.0*rand()/RAND_MAX-0.5)));
            momentumColumn.push_back(0.2*((1.0*rand()/RAND_MAX-0.5)));
            firingColumn.push_back(rand()%2);
            deltaColumn.push_back(0.01*(1.0-2.0*rand()/RAND_MAX));
            relativeBehaveColumn.push_back(0.0);

            if((i >= numInputs && i < numInputs+numOutputs) || (i == (weightsActive.size()-1))) neuronType = -1;
            if((j >= numInputs && j < numInputs+numOutputs) || (j == (weightsActive.size()-1))) neuronType = -1;

            maskV.push_back(false);
            type.push_back(neuronType);
        }
        mask.push_back(maskV);
        synapseType.push_back(type);
        relativeBehaviour.push_back(relativeBehaveColumn);
        firingMatrixCounter.push_back(firingColumn);
        weightsActive.push_back(weightColumnActive);
        biasesActive.push_back(weightColumnInactive);
        deltaMatrix.push_back(deltaColumn);
        weightsNeurons.push_back(1.0);
        //weightsNeurons[i] = 10.0*((2.0*rand()/RAND_MAX)-1.0);
        momentum.push_back(momentumColumn);
        slope.push_back(2.0*((1.0*rand()/RAND_MAX-0.5)));
        error.push_back(0.02*((1.0*rand()/RAND_MAX-0.5)));
        lastError.push_back(0.01*((1.0*rand()/RAND_MAX-0.5)));
        beforeLasteError.push_back(0.01*((1.0*rand()/RAND_MAX-0.5)));
        derivedError.push_back(0.0);
        mean.push_back(0.5);
        meanChanging.push_back(0.5);

        samplerRealInput.push_back(0.0);
        samplerRealOutput.push_back(0.0);

        samplerCounterInput.push_back(0.0);
        samplerCounterOutput.push_back(0.0);

        samplerRealEnergyBillance.push_back(0.0);
        samplerCounterEnergyBillance.push_back(0.0);

        samplerRealInputSignal.push_back(0.0);
        samplerRealOutputSignal.push_back(0.0);
        samplerCounterInputSignal.push_back(0.0);
        samplerCounterOutputSignal.push_back(0.0);



        OutputSamplerReal.push_back(1.0);
        OutputSamplerCounter.push_back(1.0);
        integratorReal.push_back(0.0);
        integratorCounter.push_back(0.0);
        OutputIntegratorReal.push_back(0.0);
        OutputIntegratorCounter.push_back(0.0);
        EnergyFlowReal.push_back(1.0);
        EnergyFlowCounter.push_back(1.0);
        ActivityReal.push_back(1.0);
        ActivityCounter.push_back(1.0);
        beforelastReal.push_back(0.0);
        beforelastCounter.push_back(0.0);
        approxError.push_back(0.0);
        outputError.push_back(0.0);
        inputError.push_back(0.0);
        globMean.push_back(relativeBehaveColumn);
        globDistErr0.push_back(relativeBehaveColumn);
        globDistErr1.push_back(relativeBehaveColumn);
        impulseResponse.push_back(0.0);
        emptyResponse.push_back(0.0);

        integrationTime = 1;
        float rnd = 1.0*rand()/RAND_MAX;
        if(i > numInputs+numOutputs) integrationSpeed.push_back(sqrt(rnd));
        else integrationSpeed.push_back(0.05);
    }

    fireCounter[fireCounter.size()-1] = 1.0;
    fireReal[fireReal.size()-1] = 1.0;

    for(int i = 0; i < weightsActive.size(); i++){
        for(int j = 0; j < weightsActive.size()-1; j++){
            if(mask[i][j] == false) mask[j][i] = (rand()%3) == true;
            if(mask[j][i] == false) mask[i][j] = (rand()%3) == true;
        }
    }


}

vector<vector<float>> NeuralCluster::getWeights(){

    return weightsActive;
}

void NeuralCluster::resetSampler(bool randomize){
        samples = 0.0;

        for(int i = 0; i < weightsActive.size()-1; i++){

            integratorCounter[i] = 0.5;
            OutputIntegratorCounter[i] = 0.5;
            integratorReal[i] = 0.5;
            OutputIntegratorReal[i] = 0.5;

            ActivityReal[i] = 0.5;
            ActivityCounter[i] = 0.5;

            EnergyFlowReal[i] = 1.0*rand()/RAND_MAX;
            EnergyFlowCounter[i] = 0.5;

            mean[i] = 0.5;
            //meanChanging[i] = 0.5;

            fireCounter[i] = 0.5;
            samplerRealInput[i]=(0.0);
            samplerRealOutput[i]=(0.0);

            samplerCounterInput[i]=(0.0);
            samplerCounterOutput[i]=(0.0);

            samplerRealEnergyBillance[i]=(0.0);
            samplerCounterEnergyBillance[i]=(0.0);

            samplerRealInputSignal[i]=(0.0);
            samplerRealOutputSignal[i]=(0.0);
            samplerCounterInputSignal[i]=(0.0);
            samplerCounterOutputSignal[i]=(0.0);
            fireReal[i] = 0.5;

            lastReal[i] = 0.5;
            lastCounter[i] = 0.5;

            beforelastReal[i] = 0.5;
            beforelastCounter[i] = 0.5;
        }

}

void NeuralCluster::pairFromTriple(float learningRate){


    for(int s = 0; s < 128; s++){

    vector<int> triple;
    triple.push_back(rand()%weightsActive.size());
    triple.push_back(rand()%weightsActive.size());
    triple.push_back(rand()%weightsActive.size());

    vector<float> comparison;
    for(int i = 0; i < triple.size(); i++){
        float comparison_factor;
        for(int j = 0; j < triple.size(); j++){
            if(i != j) comparison_factor += (relativeBehaviour[i][j]); //Negative if behaviour is inverted
        }
        comparison.push_back(comparison_factor);
    }

    float minBehave = INFINITY;
    int indexMinBehave = 0;
    for(int i = 0; i < triple.size(); i++){
        if(comparison[i] < minBehave){
            minBehave = comparison[i];
            indexMinBehave = i;
        }
    }

    weightsActive[triple[indexMinBehave]][triple[(indexMinBehave+1)%triple.size()]] -= learningRate;
    weightsActive[triple[indexMinBehave]][triple[(indexMinBehave+2)%triple.size()]] -= learningRate;

    }


}

void NeuralCluster::propergateEmpty(int steps){


    for(int i = 0; i < weightsActive.size()-1; i++) impulseResponse[i] = 0.0;

    for(int n = 0; n < weightsActive.size()-1; n++){

    this->resetSampler(false);

    for(int k = 0; k < steps; k++){

        lastReal[n]  = 0.0;

        for(int i = 0; i < weightsActive.size()-1; i++){
            float signalReal = 0.0;
            for(int j = 0; j < weightsActive.size(); j++){
                signalReal += lastReal[j]*weightsActive[i][j];
            }
            EnergyFlowReal[i] = minMax(signalReal);
        }

        beforelastReal = lastReal;
        lastReal = EnergyFlowReal;

    }

    for(int k = 0; k < steps; k++){

        lastReal[n]  = 1.0;

        for(int i = 0; i < weightsActive.size()-1; i++){
            float signalReal = 0.0;
            float differenceSig = 0.0;
            for(int j = 0; j < weightsActive.size(); j++){
                signalReal += lastReal[j]*weightsActive[i][j];
                differenceSig += (lastReal[j]-beforelastReal[j])*weightsActive[i][j];
            }
            if(i == n) impulseResponse[n] += differenceSig;
            EnergyFlowReal[i] = minMax(signalReal);
        }
        beforelastReal = lastReal;
        lastReal = EnergyFlowReal;

    }

    }

    float multiplicator = 0.0;
    for(int k = 0; k < steps; k++) multiplicator += 1.0;
    for(int i = 0; i < weightsActive.size()-1; i++) impulseResponse[i] /= multiplicator;
}


void NeuralCluster::applyLearning(float learningRate,float globalRMSError, int type){

    //Correct each neuron random independently
    vector<bool> alreadyDone;
    for(int i = 0; i < weightsActive.size(); i++)alreadyDone.push_back(false);
    for(int m = 0; m < weightsActive.size()-1; m++){

        //Select a random neuron which is not already corrected
        int i = -1;
        bool done = false;
        while(!done){
                i = rand()%(weightsActive.size()-1);
                if(alreadyDone[i] == false){
                    alreadyDone[i] = true;
                    done = true;
                }
        }

        //Calculate the negative value of the in and output signal
        float meanOutputInactive = 0.0;
        float meanInputSignal = 0.0;
        float balanceSignal = 0.0;
        float meanOutput = 0.0;
        float meanInputEnergy = 0.0;

        for(int j = 0; j < weightsActive.size()-1; j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);

            meanOutput += (activationJ)*(weightsActive[j][i]);
            balanceSignal += 2.0*(activationJ-0.5)*weightsActive[i][j];
            meanInputSignal +=  lastReal[j]*(weightsActive[i][j]);

            meanOutputInactive += (weightsActive[j][i]);
            meanInputEnergy +=  lastReal[j]*abs(weightsActive[j][i]);
        }
        float activationI = (EnergyFlowReal[i]);
        //Do the correction on the weights accourding to the current activation on it
        for(int j = 0; j < weightsActive.size()-1; j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);

            float meanOutputConter = 0.0;
            for(int k = 0; k < weightsActive.size()-1; k++) meanOutputConter += weightsActive[k][j];
            weightsActive[i][j] -= lastReal[i]*(((1.0-activationJ)*meanInputSignal+(lastReal[j])*meanOutputConter)/(weightsActive.size()))*learningRate*globalRMSError;
            //weightsActive[i][j] -= (1.0-activationJ)*(1.0-lastReal[i])*(meanInputSignal/weightsActive.size())*learningRate*globalRMSError;
            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[i][j] -= (activationJ)*(1.0-lastReal[i])*(meanInputSignal/weightsActive.size())*learningRate*globalRMSError;
            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[j][i] -= (1.0-activationJ)*lastReal[i]*(meanInputSignal/weightsActive.size())*learningRate*globalRMSError;

            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[j][i] -= (1.0-activationJ)*lastReal[i]*(meanInputSignal/weightsActive.size())*learningRate*globalRMSError;

            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[j][i] -= (activationI*(lastReal[j])*meanOutputInactive/weightsActive.size())*learningRate*globalRMSError;
            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[i][j] -= activationI*(1.0-activationI)*((abs(activationI-lastReal[i]))*(2.0*rand()/RAND_MAX-1.0))*learningRate*globalRMSError;
        }
        //TODO: Think about the bias !
        weightsActive[i][weightsActive.size()-1] -= activationI*(1.0-activationI)*(meanInputSignal/weightsActive.size()+2.0*weightsActive[i][weightsActive.size()-1])*learningRate*globalRMSError;
    }

    float sumAbsWeights = 0.0;
    //Normalize the inputs and outputs of each neuron independently by random
    for(int i = 0; i < weightsActive.size(); i++)alreadyDone[i] = (false);
    for(int m = 0; m < weightsActive.size()-1; m++){

        //Select a random neuron which is not already corrected
        int i = -1;
        bool done = false;
        while(!done){
                i = rand()%(weightsActive.size()-1);
                if(alreadyDone[i] == false){
                    alreadyDone[i] = true;
                    done = true;
                }
        }

        //Normalize the inputs and outputs of each neuron so their absoulte sum is one
        for(int j = 0; j < weightsActive.size()-1; j++){
            //weightsActive[i][j] = ((weightsActive[i][j])/(absWeightsIn))*(weightsActive.size())*1.0;

            //Switch of some weights which are not nescessary
            //if((i >= 0)&& (j >= 0) && (i < numInputs)&& (j < numInputs)){ weightsActive[i][j] = 0.0; }
            //if((i >= 0)&& (j >= 0) && (i < numInputs+numOutputs)&& (j < numInputs)){ weightsActive[i][j] = 0.0; }
            if(i != j && (i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (i < weightsActive.size())&& (j < weightsActive.size()-1)){ weightsActive[i][j] = 0.0; }
            //if((i >= (weightsActive.size()-1))&& (j >= 0) && (i <= weightsActive.size())){ weightsActive[i][j] = 0.0; }
            //if(mask[i][j]){ weightsActive[i][j] = 0.0; }

            if(i == j){ weightsActive[i][j] = 0.0; }
        }

        //Calculate it's absolute weights at input and output
        float absWeightsOut = 0.0;
        float absWeightsIn = 0.0;
        for(int j = 0; j < weightsActive.size()-1; j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);
            absWeightsOut += abs(weightsActive[j][i]);
            absWeightsIn += abs(weightsActive[i][j]);
        }

        sumAbsWeights += absWeightsIn;


        //Normalize the inputs and outputs of each neuron so their absoulte sum is one
        for(int j = 0; j < weightsActive.size()-1; j++){
            weightsActive[i][j] = ((weightsActive[i][j])/(absWeightsIn))*(weightsActive.size())*1.0;
            weightsActive[j][i] = ((weightsActive[j][i])/(absWeightsOut))*(weightsActive.size())*1.0;
        }
    }
    lastGlobErr = globalRMSError;
}

void NeuralCluster::resetDeltaMatrix(){

    for(int i = 0; i < weightsActive.size(); i++){
        for(int j = 0; j < weightsActive.size(); j++){
            deltaMatrix[i][j] = 0.0;
        }
    }
}

void NeuralCluster::trainBP(vector<float> target,float learningRate,int iterations){

}

float NeuralCluster::signum(float x){
    return 2.0/(1.0+exp(-x))-1.0;
}

vector<float> NeuralCluster::getActivation(){
    return EnergyFlowReal;
}

vector<float> NeuralCluster::getTarget(){
    return EnergyFlowReal;
}

float NeuralCluster::minMax(float x){
    return (1.0/(1.0+(exp(-(x)))));
}

void NeuralCluster::inputData(vector<float> input,vector<float> output){

    for(int i = 0; i < input.size(); i++){
        fireCounter[i] = input[i];
        EnergyFlowCounter[i] = input[i];
        ActivityCounter[i] = input[i];

        fireReal[i] = input[i];
        EnergyFlowReal[i] = input[i];
        ActivityReal[i] = input[i];
    }
    for(int i = input.size(); i < output.size()+input.size(); i++) {
        EnergyFlowReal[i] = output[i-input.size()];
        fireReal[i] = output[i-input.size()];
        ActivityReal[i] = output[i-input.size()];
    }
}

void NeuralCluster::propergate(vector<float> input,vector<float> output, float energy){


    beforelastCounter = lastCounter;
    beforelastReal = lastReal;

    lastCounter = EnergyFlowCounter;
    lastReal = EnergyFlowReal;


    for(int i = numInputs; i < numInputs+numOutputs+numHiddens+numRekurrent; i++){
        fireCounter[i] = EnergyFlowCounter[i];
    }


    for(int i = numInputs; i < numInputs+numOutputs+numHiddens+numRekurrent; i++){

        fireReal[i] = EnergyFlowReal[i];
    }


    inputData(input,output);

    vector<float> deltaEnergysReal;
    vector<float> deltaEnergysCounter;
    vector<float> deltaError;


    float absEnergyReal = 0.0;
    float absEnergyCounter = 0.0;

    for(int i = 0; i < weightsActive.size()-1; i++){
        for(int j = 0; j < weightsActive.size(); j++){
            absEnergyCounter += abs((weightsActive[i][j]+deltaMatrix[i][j])*fireCounter[j]);
        }
    }

    for(int i = 0; i < weightsActive.size()-1; i++){
        for(int j = 0; j < weightsActive.size(); j++){
            absEnergyReal += abs((weightsActive[i][j]+deltaMatrix[i][j])*fireReal[j]);
        }
    }

    float sumCounterClassyfier = 0;

        for(int i = 0; i < weightsActive.size()-1; i++){

            //cout << slope[i] << "\n";

            float EnergyInputReal = 0.0;
            float EnergyInputCounter = 0.0;

            float EnergyOutputReal = 0.0;
            float EnergyOutputCounter = 0.0;


            float InputSignalReal = 0.0;
            float InputSignalCounter = 0.0;

            float OutputSignalReal = 0.0;
            float OutputSignalCounter = 0.0;

            float errorSignal = 0.0;
            float selfSignal = 0.0;

            float inPatternAnomaly = 0.0;


            float weightsMean = 0.0;

            for(int j = 0; j < weightsActive[i].size(); j++){
                float activationI = fireReal[i];
                float activationJ = (fireReal[j]);

                if(weightsActive[i][j] < 0.0) inPatternAnomaly += (activationJ)*abs(weightsActive[i][j])*activationI;
                else inPatternAnomaly += (1.0-activationJ)*abs(weightsActive[i][j])*activationI;

                if(weightsActive[i][j] < 0.0) inPatternAnomaly -= (1.0-activationJ)*abs(weightsActive[i][j])*activationI;
                else inPatternAnomaly -= (activationJ)*abs(weightsActive[i][j])*activationI;
/*
                if(weightsActive[j][i] < 0.0) inPatternAnomaly += (activationJ)*abs(weightsActive[j][i])*activationI;
                else inPatternAnomaly += (1.0-activationJ)*abs(weightsActive[j][i])*activationI;

                if(weightsActive[j][i] < 0.0) inPatternAnomaly -= (1.0-activationJ)*abs(weightsActive[j][i])*activationI;
                else inPatternAnomaly -= (activationJ)*abs(weightsActive[j][i])*activationI;
*/


                    InputSignalCounter +=  fireCounter[j]*(weightsActive[i][j]+deltaMatrix[i][j]);//*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireCounter[j]-fireCounter[i]));
                    InputSignalReal += fireReal[j]*(1.0-0.0*fireReal[i]*(weightsActive[i][j]+deltaMatrix[i][j]))*(weightsActive[i][j]+deltaMatrix[i][j]);//*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireReal[j]-fireReal[i]));
                    selfSignal += fireReal[j]*minMax(-16.0*lastReal[i]*(weightsActive[j][i]))*(weightsActive[i][j]);

                    relativeBehaviour[i][j] = (123.0*relativeBehaviour[i][j]+(((EnergyFlowReal[i]-0.5)*(EnergyFlowReal[j]-0.5))))/124.0;


                    OutputSignalCounter +=  fireCounter[i]*(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireCounter[j]-fireCounter[i]));
                    if(weightsActive[i][j]>0.0)OutputSignalReal += fireReal[i]*abs(weightsActive[i][j]+deltaMatrix[i][j])*(fireReal[j]-0.5);//*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireReal[j]-fireReal[i]));
                    else OutputSignalReal += fireReal[i]*abs(weightsActive[i][j]+deltaMatrix[i][j])*(0.5-fireReal[j]);

                    EnergyInputCounter +=  fireCounter[j]*abs(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireCounter[j]-fireCounter[i]));
                    EnergyInputReal += fireReal[j]*(1.0-0.0*fireReal[i]*(weightsActive[i][j]+deltaMatrix[i][j]))*abs(weightsActive[i][j]+deltaMatrix[i][j]);//*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireReal[j]-fireReal[i]));

                    EnergyOutputReal +=  (fireReal[i])*abs(weightsActive[i][j]+deltaMatrix[i][j]);//*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size();
                    EnergyOutputCounter += (fireCounter[i])*abs(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size();

                    errorSignal += error[j]*weightsActive[i][j];
            }

            samplerCounterInput[i] = EnergyInputCounter;
            samplerRealInput[i] = EnergyInputReal;

            samplerCounterOutput[i] = EnergyOutputCounter;
            samplerRealOutput[i] = EnergyOutputReal;


            samplerCounterOutputSignal[i] = OutputSignalCounter;
            samplerRealOutputSignal[i] = OutputSignalReal/(0.0001+EnergyOutputReal);

            samplerCounterInputSignal[i] = InputSignalCounter;
            samplerRealInputSignal[i] = 24.0*(InputSignalReal/EnergyInputReal-0.0*beforelastReal[i]*(impulseResponse[i])/weightsActive.size())*(1.0-energy);

            samplerRealEnergyBillance[i] = samplerRealInput[i]-samplerRealOutput[i];
            float gradient = 1.0*(4.0*minMax(samplerRealInputSignal[i])*minMax(-samplerRealInputSignal[i])-4.0*EnergyFlowReal[i]*(1.0-EnergyFlowReal[i]));
            //EnergyFlowReal[i] =  (minMax(samplerRealInputSignal[i])*minMax(-gradient*8.0)+minMax(gradient*8.0)*EnergyFlowReal[i])/(minMax(-gradient*8.0)+minMax(gradient*8.0));
            if(energy > 0.0) EnergyFlowReal[i] =  (minMax(samplerRealInputSignal[i])*minMax(-gradient*5.0)*(energy)+minMax(gradient*5.0)*EnergyFlowReal[i]*(1.0-energy))/(minMax(-gradient*5.0)*(energy)+minMax(gradient*5.0)*(1.0-energy));
            else  EnergyFlowReal[i] =  (minMax(samplerRealInputSignal[i])*minMax(-gradient*5.0)+minMax(gradient*5.0)*EnergyFlowReal[i])/(minMax(-gradient*5.0)+minMax(gradient*5.0));
            EnergyFlowReal[i] = EnergyFlowReal[i]+1.0*EnergyFlowReal[i]*(1.0-EnergyFlowReal[i])*((2.0*rand()/RAND_MAX)-1.0)*(energy);

            meanChanging[i] = (42.0*meanChanging[i]-(meanChanging[i]-EnergyFlowReal[i]))/(43.0);

            EnergyFlowReal[i] *= 1.5-1.5*minMax(inPatternAnomaly);
            EnergyFlowReal[i] *= 2.0/3.0;
            //ActivityReal[i] = (fireReal[i]+samples*ActivityReal[i])/(samples+1.0);
            //EnergyFlowReal[i] = 0.5*(weightsActive.size())*samplerReal[i]/energyAbs;

        }

        for(int i = input.size()+output.size(); i < weightsActive.size()-1; i++){
        }

        for(int i = numInputs; i < weightsActive.size()-1; i++){
            //samplerCounter[i] = minMax(integratorCounter[i]-OutputIntegratorCounter[i]);//+samples*samplerCounter[i])/(samples+1.0);
            //OutputSamplerCounter[i] = (minMax(OutputIntegratorCounter[i])+samples*OutputSamplerCounter[i])/(samples+1.0);

            //ActivityCounter[i] = (fireCounter[i]+samples*ActivityCounter[i])/(samples+1.0);
            //EnergyFlowCounter[i] = 0.5*(weightsActive.size())*samplerCounter[i]/energyAbs;
            samplerCounterEnergyBillance[i] = samplerCounterInput[i]-samplerCounterOutput[i];
            EnergyFlowCounter[i] =  minMax(samplerCounterInputSignal[i]);
        }

        inputData(input,output);

        for(int i = 0; i < weightsActive.size()-1; i++) mean[i] = (128.0*mean[i]+EnergyFlowReal[i])/(129.0);

        //if(hiddenWrite) for(int i = 0; i < input.size(); i++) samplerCounter[numInputs+numOutputs+numHiddens+i] = (samplerCounter[numInputs+numOutputs+numHiddens+i])*input[i];
        //if(hiddenWrite) for(int i = 0; i < input.size(); i++) samplerReal[numInputs+numOutputs+numHiddens+i] = (samplerReal[numInputs+numOutputs+numHiddens+i])*input[i];


        //samples++;
}
