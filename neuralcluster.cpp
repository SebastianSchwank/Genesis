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


            type.push_back(neuronType);
        }
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
    }

    fireCounter[fireCounter.size()-1] = 1.0;
    fireReal[fireReal.size()-1] = 1.0;



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

            EnergyFlowReal[i] = 0.5;
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
            if(i == n) impulseResponse[n] += differenceSig*exp(-k*(1.0/steps));
            EnergyFlowReal[i] = minMax(signalReal);
        }
        beforelastReal = lastReal;
        lastReal = EnergyFlowReal;

    }

    }


    for(int i = 0; i < weightsActive.size()-1; i++) impulseResponse[i] /= steps;
}

void NeuralCluster::propergateImpulse(int steps, vector<float> input, vector<float> output){



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


    for(int i = 0; i < weightsActive.size()-1; i++) impulseResponse[i] /= steps;

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
        float meanOutput = 0.0;
        float meanInput = 0.0;
        float meanOutputInactive = 0.0;
        float meanInputSignal = 0.0;
        float meanActivity = 0.0;
        float absDeviation = 0.0;
        float meanInputInactive = 0.0;

        float selfSignal = 0.0;

        for(int j = 0; j < weightsActive.size(); j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);

             meanOutput += (activationJ)*(weightsActive[j][i]);



            if(weightsActive[i][j] < 0.0) meanInput += (activationJ)*(weightsActive[i][j]);
            else meanInput += (1.0-activationJ)*(weightsActive[i][j]);
            meanInputSignal +=  activationJ*(weightsActive[i][j]);

            meanOutputInactive += (weightsActive[j][i]);
            meanInputInactive +=  (weightsActive[i][j]);
            meanActivity += (activationJ);

            absDeviation += abs(activationJ-mean[j])*(weightsActive[i][j]);

            selfSignal += activationJ*minMax(16.0*activationI*(weightsActive[j][i]))*(weightsActive[i][j]);
        }
        //absDeviation /= weightsActive.size();

        float activationI = minMax(meanInputSignal)*EnergyFlowReal[i];
        float gradientI = minMax(meanInputSignal)*minMax(-meanInputSignal);
        float coresspondenceIn = 0.0;
        float coresspondenceOut = 0.0;

        for(int j = 0; j < weightsActive.size()-1; j++){
            float activationJ = (EnergyFlowReal[j]);


            if(weightsActive[i][j] < 0.0) coresspondenceIn += (activationJ)*abs(weightsActive[i][j])*activationI;
            else coresspondenceIn -= (activationJ)*abs(weightsActive[i][j])*activationI;

            if(weightsActive[j][i] < 0.0) coresspondenceOut += (0.5-activationJ)*(weightsActive[j][i])*activationI;
            else coresspondenceOut += (activationJ-0.5)*(weightsActive[j][i])*activationI;


            //if(weightsActive[j][i] < 0.0) coresspondence += (activationJ)*(weightsActive[j][i])*(0.5-activationI);
            //else coresspondence += (activationJ)*(weightsActive[j][i])*(activationI-0.5);


            //if(weightsActive[i][j] < 0.0) coresspondence += (1.0-activationJ)*(weightsActive[i][j])*activationI;
            //else coresspondence -= (1.0-activationJ)*(weightsActive[i][j])*activationI;

        }

        //coresspondence /= weightsActive.size();

        //Do the correction on the weights accourding to the current activation on it
        for(int j = 0; j < weightsActive.size()-1; j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);

            float gradient = minMax((meanInputSignal-0.0*lastReal[i]*(impulseResponse[i]-emptyResponse[i]))/weightsActive.size())*minMax(-(meanInputSignal-0.0*lastReal[i]*(impulseResponse[i]-emptyResponse[i]))/weightsActive.size());


            //weightsActive[i][j] -= weightsActive[i][j]*(activationJ)*(activationI)*(0.25-abs((activationI-0.5)*(activationJ-0.5)))*learningRate;

                //weightsActive[i][j] += activationI*(activationI-mean[i])*(activationJ-mean[j])*abs(mean[i]-0.5)*learningRate;
                //weightsActive[j][i] -= (activationJ)*abs(activationJ-mean[i])*(activationI-mean[i])*learningRate;

            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[i][j] += (coresspondenceOut)/weightsActive.size()*activationI*learningRate*globalRMSError;

            if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[i][j] += (coresspondenceIn+coresspondenceOut)/weightsActive.size()*gradient*learningRate*globalRMSError;

            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[j][i] -= (coresspondence)/weightsActive.size()*gradient*learningRate*globalRMSError;

            //if(type == synapseType[j][i]||type == synapseType[i][j] || synapseType[i][j] == -1 || synapseType[j][i] == -1) weightsActive[i][j] -= meanChanging[j]*meanChanging[i]*learningRate*minMax(-16.0*relativeBehaviour[i][j])*minMax(16.0*relativeBehaviour[i][j])*(globalRMSError);
            if(type == synapseType[i][j]||type == synapseType[j][i] || synapseType[j][i] == -1 || synapseType[i][j] == -1) weightsActive[j][i] -= (activationJ)*activationI*gradient*((meanOutputInactive)/weightsActive.size())*learningRate*(globalRMSError);

                //weightsActive[i][j] -= (activationJ)*(activationI)*((meanInputInactive)/weightsActive.size())*learningRate;
                //weightsActive[j][i] -= ((activationI))*((meanOutput)/weightsActive.size())*learningRate;



            //weightsActive[i][j] -= (1.0-activationJ)*activationI*(meanInput/weightsActive.size())*((1.0-minMax(mean[i]*weightsActive[j][i]-mean[j]*weightsActive[i][j]))*minMax(mean[i]*weightsActive[j][i]-mean[j]*weightsActive[i][j]))*learningRate;
            //weightsActive[i][j] -= (activationJ)*(1.0-activationI)*(((((((meanInput/weightsActive.size())))))))*learningRate;


            //weightsActive[j][i] -= (activationJ)*(activationI)*(meanOutputInactive/weightsActive.size())*((1.0-minMax(mean[j]*weightsActive[i][j]-mean[i]*weightsActive[j][i]))*minMax(mean[j]*weightsActive[i][j]-mean[i]*weightsActive[j][i]))*learningRate;

            //weightsActive[i][j] -= activationJ*(1.0-activationJ)*abs(0.5-activationI)*weightsActive[i][j]*learningRate;
            //weightsActive[j][i] -= activationJ*(1.0-activationJ)*weightsActive[i][j]*learningRate;
            //weightsActive[i][j] += abs(0.5-activationJ)*(activationI)*(1.0-activationI)*weightsActive[i][j]*learningRate;

            //weightsActive[j][i] -= activationJ*(1.0-activationJ)*abs(0.5-activationI)*weightsActive[j][i]*learningRate;

            //weightsActive[i][j] += (activationI)*(((((log(1.0+exp(-(activationI)*(meanOutput)/weightsActive.size())))))))*learningRate;
            //weightsActive[j][i] -= (activationJ)*(((((log(1.0+exp((activationI)*meanInputInactive/weightsActive.size())))))))*learningRate;


            //weightsActive[j][i] += (activationI)*((((((0.0+exp(-(activationI)*(meanInput)/weightsActive.size())))))))*learningRate;
            //weightsActive[i][j] -= (activationJ)*((((((0.0+exp((activationI)*meanOutputInactive/weightsActive.size())))))))*learningRate;

            //weightsActive[i][j] -= activationI*activationJ*(activationJ*weightsActive[i][j]+activationI*weightsActive[j][i])*learningRate;
            //weightsActive[i][j] += activationI*activationJ*(2.0*rand()/RAND_MAX-1.0)*learningRate;
        }

        //float activationI = (EnergyFlowReal[i]);
        weightsActive[i][weightsActive.size()-1] -= (meanInputSignal)/weightsActive.size()*learningRate;
    }

    float sumAbsWeights = 0.0;
    //Normalize the inputs and outputs of each neuron independently by random
    for(int i = 0; i < weightsActive.size(); i++)alreadyDone[i] = (false);
    for(int m = 0; m < weightsActive.size(); m++){

        //Select a random neuron which is not already corrected
        int i = -1;
        bool done = false;
        while(!done){
                i = rand()%(weightsActive.size());
                if(alreadyDone[i] == false){
                    alreadyDone[i] = true;
                    done = true;
                }
        }

        //Calculate it's absolute weights at input and output
        float absWeightsOut = 0.0;
        float absWeightsIn = 0.0;
        for(int j = 0; j < weightsActive.size(); j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);
            absWeightsOut += abs(weightsActive[j][i]);
            absWeightsIn += abs(weightsActive[i][j]);
        }

        sumAbsWeights += absWeightsIn;


        //Normalize the inputs and outputs of each neuron so their absoulte sum is one
        for(int j = 0; j < weightsActive.size(); j++){
            weightsActive[j][i] = ((weightsActive[j][i])/(absWeightsOut))*weightsActive.size()*1.0;
            weightsActive[i][j] = ((weightsActive[i][j])/(absWeightsIn))*weightsActive.size()*1.0;

            //Switch of some weights which are not nescessary
            //if((i >= 0)&& (j >= 0) && (i < numInputs)&& (j < weightsActive.size())){ weightsActive[i][j] = 0.0; }
            //if((i >= 0)&& (j >= 0) && (i < numInputs+numOutputs)&& (j < numInputs)){ weightsActive[i][j] = 0.0; }
            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (i < weightsActive.size()-16)&& (j < weightsActive.size()-1)){ weightsActive[i][j] = 0.0; }
            //if((i >= (weightsActive.size()-16))&& (j >= 0) && (i < weightsActive.size())&& (j < numInputs+numOutputs)){ weightsActive[i][j] = 0.0; }

            if(i == j){ weightsActive[i][j] = 0.0; }
        }
    }
    for(int i = 0; i < weightsActive.size(); i++){
    for(int j = 0; j < weightsActive.size(); j++){
        //weightsActive[i][j] = ((weightsActive[i][j])/sumAbsWeights)*weightsActive.size()*weightsActive.size();
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

    //float signum = 1.0;
    //if(x < 0.0) x = 0.0;
    //if(x >= 0.0) signum = 1.0;


    //if( x > 0.0) x = (2.0/(1.0+(exp(-x))))-1.0;
    //else x = 0.0;
    //return x;

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

    /*
    if(inverted){
        for(int i = 0; i < input.size(); i++){ samplerReal[i] = input[i];}
        for(int i = numInputs; i < output.size()+numInputs; i++) { samplerCounter[i] = output[i-numInputs]; samplerReal[i] = output[i-numInputs]; }
    }
    */


    for(int i = numInputs; i < numInputs+numOutputs+numHiddens+numRekurrent; i++){

        //integratorCounter[i] *= 0.9;
        fireCounter[i] = EnergyFlowCounter[i];

        //fireCounter[i] = 0.0;
        //if(samplerCounter[i] > ((2.0*rand()/RAND_MAX)-1.0)) fireCounter[i] = 1.0;
        /*
        float probability = 1.0-(1.0/(1.0+(exp(-EnergyFlowCounter[i]))));
        float ActivationProbaility = 1.0*rand()/RAND_MAX;
        if(probability < ActivationProbaility) fireCounter[i] = 1.0;
        else fireCounter[i] = 0.0;
        */
    }


    for(int i = numInputs; i < numInputs+numOutputs+numHiddens+numRekurrent; i++){


        //integratorReal[i] *= 0.9;

        //fireReal[i] = 0.0;
        //if(samplerReal[i] > ((2.0*rand()/RAND_MAX)-1.0)) fireReal[i] = 1.0;

        fireReal[i] = EnergyFlowReal[i];

/*
        float probability = 1.0-(1.0/(1.0+(exp(-EnergyFlowReal[i]))));
        float ActivationProbaility = 1.0*rand()/RAND_MAX;
        if(probability < ActivationProbaility) fireReal[i] = 1.0;
        else fireReal[i] = 0.0;
*/
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


            float weightsMean = 0.0;

            for(int j = 0; j < weightsActive[i].size(); j++){



                    InputSignalCounter +=  fireCounter[j]*(weightsActive[i][j]+deltaMatrix[i][j]);//*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireCounter[j]-fireCounter[i]));
                    InputSignalReal += fireReal[j]*(weightsActive[i][j]+deltaMatrix[i][j]);//*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireReal[j]-fireReal[i]));
                    selfSignal += fireReal[j]*minMax(-16.0*lastReal[i]*(weightsActive[j][i]))*(weightsActive[i][j]);

                    relativeBehaviour[i][j] = (123.0*relativeBehaviour[i][j]+(((EnergyFlowReal[i]-0.5)*(EnergyFlowReal[j]-0.5))))/124.0;


                    OutputSignalCounter +=  fireCounter[i]*(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireCounter[j]-fireCounter[i]));
                    OutputSignalReal += fireReal[i]*(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireReal[j]-fireReal[i]));

                    EnergyInputCounter +=  fireCounter[j]*abs(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireCounter[j]-fireCounter[i]));
                    EnergyInputReal += fireReal[j]*abs(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireReal[j]-fireReal[i]));

                    EnergyOutputReal +=  (fireReal[i])*abs(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size();
                    EnergyOutputCounter += (fireCounter[i])*abs(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size();

                    errorSignal += error[j]*weightsActive[i][j];
                    /*
                    if(fireCounter[j] == 1.0) EnergyCounter +=  (weightsActive[i][j])*(lastCounter[j]);
                    else EnergyCounter +=  (weightsInactive[i][j])*(lastCounter[j]);
                    if(fireReal[j] == 1.0)EnergyReal +=  (weightsActive[i][j])*(lastReal[j]);
                    else EnergyReal +=  (weightsInactive[i][j])*(lastReal[j]);
                    */

                     //integratorReal[j] += weights[i][j]*fireReal[i];
                     //integratorCounter[j] += weights[i][j]*fireCounter[i];


                     //EnergyReal +=  weights[i][j]*fireReal[j];
                     //EnergyCounter +=  weights[i][j]*fireCounter[j];

                     //integratorReal[j] -= weights[i][j]*(fireReal[j]);;
                     //integratorCounter[j] -= weights[i][j]*(fireCounter[j]);


                     //samplerReal[j] -= deltaEnergyReal;
                     //samplerCounter[j] -= deltaEnergyCounter;


                     //samplerReal[j] -= deltaEnergyReal;
                     //samplerCounter[j] -= deltaEnergyCounter;

                     //if((i > numInputs-1)&& (i < numInputs+numOutputs)) cout << (fireReal[j]-lastReal[j]) << ",";

                    /*
                     {samplerReal[j] -= deltaEnergyReal; samplerReal[i] += deltaEnergyReal;}
                     {samplerCounter[j] -= deltaEnergyCounter; samplerCounter[i] += deltaEnergyCounter;}
                    */
                     //sumWeights += minMax(weights[i][j]);
            }

            //derived[i] = cos(y);
            /*
            x = sin(x);
            y = sin(y);
            */

            samplerCounterInput[i] = EnergyInputCounter;
            samplerRealInput[i] = EnergyInputReal;

            samplerCounterOutput[i] = EnergyOutputCounter;
            samplerRealOutput[i] = EnergyOutputReal;

            samplerCounterInputSignal[i] = InputSignalCounter;
            samplerRealInputSignal[i] = InputSignalReal-1.0*lastReal[i]*(impulseResponse[i]);

            samplerCounterOutputSignal[i] = OutputSignalCounter;
            samplerRealOutputSignal[i] = OutputSignalReal;

            //error[i] += signum(errorSignal);

            //cout << "\n";


            //deltaError.push_back(deltaErr);

            //derived[i] = cos(y);

            //if(maxResultReal < abs(minMax(x))) maxResultReal = abs(minMax(x));
            //if(maxResultCounter < abs(minMax(y))) maxResultCounter = abs(minMax(y));

            //derived[i] = exp(-y);
            //if(deltaEnergys[i] == 0) derived[i] = 0.0;


            //deltaEnergysI.push_back(x/weightsSum);
            //deltaEnergys.push_back(y/weightsSum);

            //counterNetActivationDerivative[i] = cos(x);
            //slowness[i] = 1.0*rand()/RAND_MAX;
        }

        for(int i = 0; i < weightsActive.size()-1; i++){
            //samplerReal[i] = minMax(integratorReal[i]-OutputIntegratorReal[i]);//+samples*samplerReal[i])/(samples+1.0);
            //OutputSamplerReal[i] = (minMax(OutputIntegratorReal[i])+samples*OutputSamplerReal[i])/(samples+1.0);

            samplerRealEnergyBillance[i] = samplerRealInput[i]-samplerRealOutput[i];
            EnergyFlowReal[i] +=  minMax(samplerRealInputSignal[i])-lastReal[i];

            meanChanging[i] = (42.0*meanChanging[i]-(meanChanging[i]-EnergyFlowReal[i]))/(43.0);



            //ActivityReal[i] = (fireReal[i]+samples*ActivityReal[i])/(samples+1.0);
            //EnergyFlowReal[i] = 0.5*(weightsActive.size())*samplerReal[i]/energyAbs;

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
