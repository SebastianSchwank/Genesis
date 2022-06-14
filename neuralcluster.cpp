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

        }
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
            meanChanging[i] = 0.5;

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

void NeuralCluster::removeNonlin(float learningRate){

}

void NeuralCluster::train(float learningRate, float sqError){
/*
    float maxVal = 0.0;
    vector<int> neuronsChain;
    int maxChainLength = 16;

    int length = rand()%maxChainLength;

    neuronsChain.push_back(rand()%weights.size());
    for(int i = 0; i < length; i++){
        int k = rand()%weights.size();
        while(weights[neuronsChain[i]][k] == 0.0){
            k = rand()%weights.size();
        }
        neuronsChain.push_back(k);
    }

    float ErrorSum = 0.0;
    for(int i = 0; i < neuronsChain.size()-1; i++){
        ErrorSum += (realNetActivation[neuronsChain[i]]-counterActivation[neuronsChain[i]]);
    }

    for(int i = 1; i < neuronsChain.size(); i++){
        weights[neuronsChain[i-1]][neuronsChain[i]] += counterActivation[neuronsChain[i]] * ErrorSum  * counterActivation[neuronsChain[i-1]] * (1.0-counterActivation[neuronsChain[i-1]]);
    }
*/

    beforeLasteError = lastError;
    lastError = error;
    float absMax = 0.0;
    for(int i = 0; i < weightsActive.size()-1; i++){
        //if((i >= numInputs) && (i < numInputs+numOutputs+numHiddens)) error[i] = (fireReal[i]-fireCounter[i]);
        //if(i < numInputs) error[numInputs+numOutputs+numHiddens+i] = (fireReal[i]-fireCounter[i]);

        //error[i] = (EnergyFlowReal[i]-EnergyFlowCounter[i]);
        //if(i >= numInputs+numOutputs) error[i] = -error[i];
        //slope[i] += (slope[i]-(fireReal[i]-fireCounter[i])*(fireReal[i]-fireCounter[i])*fireCounter[i]*0.25)*0.001;
        //if(i > numInputs+numOutputs+numHiddens) error[i] = (fireReal[i+numInputs+numOutputs+numHiddens]-fireCounter[i+numInputs+numOutputs+numHiddens]);

        //derivedError[i] = (sqError[i])/((sqError[i]*lastError[i]));
        //if(derivedError[i] != derivedError[i]) derivedError[i] = 0.0;
        //cout << i << ":" << derivedError[i] << " ";

        float errorTerm = 0.0;


        for(int j = 0; j < weightsActive[i].size(); j++){
            //momentum[i][j] += lastCounter[j]*(realNetActivation[i]-counterActivation[i])*(counterActivation[i])*(1.0-counterActivation[i])*learningRate;
            //momentum[i][j] -= (1.0-lastCounter[j])*(realNetActivation[i]-counterActivation[i])*(counterActivation[i])*(1.0-counterActivation[i])*learningRate;


            //weights[j][i] += ((lastCounter[j])*(realNetActivation[i]-counterActivation[i]))*(1.0+counterActivation[i])*(1.0-counterActivation[i])*learningRate;
            //weights[i][j] -= ((2.0-lastCounter[j])*(realNetActivation[i]-counterActivation[i]))*(1.0+counterActivation[i])*(1.0-counterActivation[i])*learningRate;
          //weights[j][i] += ((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*(1.0+counterActivation[i])*(1.0-counterActivation[i])*learningRate;
            //weights[j][i] += ((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*counterActivation[i]*(1.0-counterActivation[i])*learningRate;
            //weights[j][i] += ((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*counterActivation[i]*(1.0-counterActivation[i])*learningRate;
            //weights[j][i] += (((counterActivation[j])*(realNetActivation[i]-counterActivation[i]))*counterActivation[i]*(1.0-counterActivation[i]))*learningRate;
            //weights[i][j] -= (1.0-(lastCounter[j]))*(realNetActivation[i]-counterActivation[i])*counterActivation[i]*(1.0-counterActivation[i])*learningRate;

            //deltaSynapse[i][j] = realNetActivation[j]*weights[i][j]*realNetActivation[i]-counterActivation[j]*weights[i][j]*counterActivation[i];

            bool skip = false;


            //if((i >= 0)&&(j >= 0) && (i < numInputs) && (j < numInputs)){ weightsActive[i][j] = 0.0; weightsInactive[i][j] = 0.0;skip = true;}

            if((i >= 0)&& (j >= 0) && (i < numInputs+numOutputs)&& (j < numInputs+numOutputs)){ weightsActive[i][j] = 0.0; biasesActive[i][j] = 0.0; skip = true;}
            //if((i >= numInputs)&& (j >= numInputs) && (i < numInputs+numOutputs)&& (j < numInputs+numOutputs)){ weights[i][j] = 0.0; skip = true;}

            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (j < numInputs+numOutputs+numHiddens) && (i < numInputs+numOutputs+numHiddens)){ weightsActive[i][j] = 0.0;weightsInactive[i][j] = 0.0; skip = true;}
            if(i == j){ weightsActive[i][j] = 0.0;biasesActive[i][j] = 0.0; skip = true;}

            //if((i >= numInputs+numOutputs)&& (j >= numInputs+numOutputs) && (j < numInputs+numOutputs+numHiddens) && (i < numInputs+numOutputs+numHiddens)){ weights[i][j] = 0.0; skip = true;}

            //if((i >= numInputs+numOutputs+numHiddens)&& (j >= numInputs+numOutputs+numHiddens) && (j <= numInputs+numOutputs+numHiddens+numRekurrent-1) && (i <= numInputs+numOutputs+numHiddens+numRekurrent)){ weights[i][j] = 0.0; skip = true;}

            //if((i >= numInputs+numOutputs+numHiddens+numRekurrent)&& (j >= numInputs+numOutputs+numHiddens+numRekurrent) && (j <= numInputs+numOutputs+numHiddens+numRekurrent*2-1) && (i <= numInputs+numOutputs+numHiddens+numRekurrent*2)){ weights[i][j] = 0.0; skip = true;}

            //if((i > numInputs+numOutputs+numHiddens/2)&& (j > numInputs+numOutputs+numHiddens/2) && (j <= numInputs+numOutputs+numHiddens-1) && (i <= numInputs+numOutputs+numHiddens)){ weights[i][j] = 0.0; skip = true;}
            //if((i > numInputs+numOutputs+numHiddens)&& (j > numInputs+numOutputs+numHiddens) && (j <= numInputs+numOutputs+numHiddens+numRekurrent-1) && (i <= numInputs+numOutputs+numHiddens+numRekurrent)){ weights[i][j] = 0.0; skip = true;}


            //if(i == j){ weights[i][j] = 0.0; skip = true;}

            //if(rand()%16 != 0) skip = true;




            //weightsActive[i][j] -= signum(weightsActive[i][j])*actJ*actI*(learningRate)*0.1*sqError;
            //weightsActive[j][i] -= signum(weightsActive[j][i])*actJ*actI*(learningRate)*0.1*sqError;

            inputError[i] *= (sqError);
            outputError[i] *= (sqError);

            //error[i] += signum(-samplerRealInputSignal[i])*(sqError);

            inputError[i] += signum(-samplerRealInputSignal[i])*(sqError);
            outputError[i] += signum(samplerRealOutputSignal[i])*(1.0-sqError);

            if(!skip){

                //weights[i][j] *= counterActivation[j]*(1.0-abs(realNetActivation[i]-counterActivation[i]))*learningRate*0.001+1.0;
                //weights[i][j] *= 1.0+counterActivation[j]*(1.0-abs(realNetActivation[i]-counterActivation[i]));



                //momentum[i][j] *= currentError;
                //momentum[i][j] *= 0.9;
                errorTerm =   (error[i]);//+(lastError[j])*(1.0-abs(error[i]))*signum(-weightsActive[j][i]))*learningRate;

                //weightsActive[i][j] = weightsActive[i][j]*(1.0-(lastReal[j]*lastCounter[j])*abs(error[j]*error[i])*learningRate);

                float signu = 1.0;


                if(i >= 0 && j >= 0){

                    float actI = (EnergyFlowReal[i]);
                    float actJ = (EnergyFlowReal[j]);

                    int k = rand()%weightsActive.size();
                    while(!(i != k && j != k)) k = rand()%weightsActive.size();


                    //weightsActive[i][j] += actJ*signum((actJ*abs(outputError[j])*weightsActive[i][j]-actI*abs(inputError[i])*weightsActive[j][i]))*weightsActive[i][j]*0.001*(sqError);
                    //weightsActive[j][i] -= signum((actJ*abs(outputError[j]*weightsActive[i][j])-actI*abs(inputError[i]*weightsActive[j][i])))*weightsActive[j][i]*0.001*(sqError);
                    //weightsActive[j][i] -= actI*signum((abs(inputError[j])-abs(outputError[i]))*weightsActive[j][i])*0.001*sqError;

                    if(sqError-lastErrorSave < 0.0){

                        weightsActive[i][j] += deltaMatrix[i][j];
                        deltaMatrix[i][j] = (1.0-actJ)*(1.0-sqError)*deltaMatrix[i][j];
                    }

                    deltaMatrix[i][j] += actJ*(1.0-2.0*rand()/RAND_MAX)*sqError*0.01;
                    deltaMatrix[i][j] -= actJ*weightsActive[i][j]*0.01;
                    //weightsActive[i][j] -= (actJ*sqError)*weightsActive[i][j]*0.0001;
                    //weightsActive[i][j] -= actI*signum(abs(inputError[j]*actI)-abs(inputError[i]))*weightsActive[i][j]*0.001*(sqError);




                    //weightsActive[j][i] += actI*signum(minMax(inputError[i])*(1.0-minMax(inputError[i])*outputError[i]))*0.001;
                    //weightsActive[j][i] -= actJ*(error[i]-error[j])*0.001*(sqError);
                    //weightsActive[j][i] -= actJ*error[i]*0.001;

/*
                    approxError[i] += signum(deltaMatrix[i][j])*actJ*actI*(1.0-actI)*0.1*sqError;
                    approxError[j] -= signum(deltaMatrix[i][j])*actJ*actI*(1.0-actI)*0.1*sqError;

                    weightsActive[i][j] -= (1.0-(error[j]))*(1.0+(error[j]))*error[j]*actI*actJ*(weightsActive[i][j])*(1.0-abs(error[i]))*abs(error[i]);
                    weightsActive[j][i] += (1.0-(error[j]))*(1.0+(error[j]))*error[j]*actI*actJ*(weightsActive[i][j])*(1.0-abs(error[i]))*abs(error[i]);
*/
                    /*
                    float errorOther = 0.0;
                    for(int k = 0; k < weightsActive.size()-1; k++){
                        float actK = (EnergyFlowCounter[k]*EnergyFlowReal[k]);
                        errorOther += ((actI*actK*(error[i])*weightsActive[i][k]*weightsActive[k][i])+(actK*(error[k])*weightsActive[i][k]));
                    }
                    */

                    //momentum[i][j] = actJ*actI*signum((actJ*weightsActive[i][j])/(1.0+samplerRealOutput[j])-((samplerRealInputSignal[i])/(1.0+samplerRealInput[i])))+momentum[i][j]*0.9;


                    //weightsActive[i][j] += (actJ)*(signum(((((samplerRealOutputSignal[i])/(1.0+samplerRealOutput[j])))-((samplerRealInputSignal[i]/(1.0+samplerRealInput[i]))))))*learningRate;

                    //weightsActive[i][weightsActive.size()-1] += (1.0-(1.0-actJ*actI))*actJ*actI*(signum(((((samplerRealOutputSignal[j])*(samplerRealEnergyBillance[j])))-((samplerRealInputSignal[i]*weightsActive[i][j]*(-samplerRealEnergyBillance[i]))))))*learningRate;

                    //weightsActive[i][j] -= actJ*actI*((((samplerRealInputSignal[j]*samplerRealOutputSignal[j]))/(1.0+samplerRealOutput[j]+samplerRealInput[j]))-(((samplerRealInputSignal[i]*samplerRealOutputSignal[i]))/(1.0+samplerRealInput[i]+samplerRealOutput[i])))*learningRate;
                    //weightsActive[i][weightsActive.size()-1] += (1.0-(1.0-actJ*actI))*actJ*actI*((((samplerRealInputSignal[j]*samplerRealOutputSignal[j]))/(1.0+samplerRealOutput[j]+samplerRealInput[j]))-(((samplerRealInputSignal[i]*samplerRealOutputSignal[i]))/(1.0+samplerRealInput[i]+samplerRealOutput[i])))*learningRate;
                    //Does rand() work ?

                    //weightsActive[j][i] -= actJ*actI*(signum(((((samplerRealOutputSignal[j]*weightsActive[i][j])/(1.0+samplerRealOutput[j]))-((samplerRealInputSignal[i])/(1.0+samplerRealInput[i]))))))*learningRate;
                    //weightsActive[i][j] -= actJ*actI*(signum(samplerRealEnergyBillance[i]*weightsActive[i][j]))*learningRate;


                    //weightsActive[i][weightsActive.size()-1] -= (actJ*actI*(1.0-(1.0-actJ*actI))*(signum((((samplerRealOutputSignal[j]*weightsActive[i][j])/(1.0+samplerRealOutput[j]))-(samplerRealInputSignal[i]/(1.0+samplerRealInput[i]))))))*learningRate;
                    /*
                    float sumMul = 0.0;
                    for(int k = numInputs+numOutputs; k < weightsActive.size(); k++){

                        if(k != i && k != j){
                            float actK = (EnergyFlowCounter[k]+EnergyFlowReal[k])*0.5;

                            weightsActive[k][j] += actK*(actI+actJ)*0.5*(1.0-2.0*rand()/RAND_MAX)*(1.0-abs(actI-actJ))*learningRate;
                            weightsActive[k][i] += actK*(actI+actJ)*0.5*(1.0-2.0*rand()/RAND_MAX)*(1.0-abs(actI-actJ))*learningRate;
                        }
                    }
                    */

                    /*
                    weightsActive[j][i] += actJ*actI*(actJ*minMax(mean[i])-actI*minMax(mean[j]))*learningRate;
                    weightsActive[i][j] -= actJ*actI*(actJ*minMax(mean[i])-actI*minMax(mean[j]))*learningRate;
                    */



                    //weightsActive[j][i] += actJ*actI*((signum((samplerRealOutputSignal[j]*weightsActive[i][j]-samplerRealInputSignal[i])))+0.1*momentum[i][j])*learningRate;

                    //weightsActive[i][weightsActive.size()-1] -= (1.0-(1.0-actJ*actI))*actJ*actI*((signum((samplerRealOutputSignal[j]*weightsActive[i][j]-samplerRealInputSignal[i])))+0.0*momentum[i][j])*learningRate;


                    //weightsActive[i][weightsActive.size()-1] += (1.0-(1.0-actJ*actI))*actJ*actI*((signum((samplerCounterOutputSignal[j]*weightsActive[i][j]-samplerCounterInputSignal[i])))+0.1*momentum[i][j])*learningRate;

                    //weightsActive[j][i] -= correcture;
                    //weightsActive[j][i] += actJ*abs(0.5-minMax((samplerCounterOutputSignal[i]-samplerCounterInputSignal[j])))*(1.0-2.0*rand()/RAND_MAX)*learningRate;
                    //weightsActive[j][i] += actJ*actI*signum((samplerCounterInputSignal[i]-samplerCounterOutputSignal[j]))*learningRate*0.1;

                    //weightsActive[i][weightsActive.size()-1] += (1.0-(1.0-actJ))*actJ*signum(samplerCounterEnergyBillance[i]-samplerCounterEnergyBillance[j])*learningRate*0.1;
                    //weightsActive[j][weightsActive.size()-1] += (1.0-(1.0-actI*actJ))*actJ*actI*signum(samplerCounterEnergyBillance[i]*samplerCounterInputSignal[i])*learningRate*weightsActive[j][i]*0.1;


                    //weightsActive[i][j] -= actI*signum((actJ-weightsActive[i][j]*actI))*(1.0-meanChanging[i])*learningRate;
                    //weightsActive[i][j] += actJ*((1.0-actJ)-weightsActive[i][j]*actI)*(1.0-meanChanging[i])*learningRate;
                    //weightsActive[i][j] -= actJ*(actJ-weightsActive[i][j]*(actI))*(1.0-meanChanging[i])*learningRate;


                    //deltaMatrix[i][j] += actJ*(2.0*rand()/RAND_MAX-1.0)*abs(error[i])*learningRate;
                    //deltaMatrix[i][j] -= actJ*abs(error[i])*abs(error[j])*signum(weightsActive[i][j])*learningRate;

                    //weightsActive[j][i] -= abs(error[i])*actJ*(error[i]+signum((actJ*(error[j])*weightsActive[i][j])))*errorDec*learningRate*0.1;
                    //weightsActive[j][i] -= actJ*(error[i]+signum((actJ*(error[j])*weightsActive[i][j])))*errorDec*learningRate;
                    //weightsActive[j][i] += actJ*signum(abs(actI*weightsActive[j][i]))*signum((error[i]*-weightsActive[i][j]))*errorDec*learningRate*signum(weightsActive[j][i]);

                    //if(((actI*sqrt(error[i]*error[i])*sqrt(weightsActive[j][i]*weightsActive[j][i])) > (actJ*sqrt(error[j]*error[j])*sqrt(weightsActive[i][j]*weightsActive[i][j]))))



/*
                        float errorBounceI = error[j];
                        float errorI = 0.0;


                        float errorBounceJ = error[i];
                        float errorJ = 0.0;


                        for(int i = 0; i < 2; i++){
                            errorI = signum(EnergyFlowReal[j]*errorBounceI*weightsActive[i][j]);
                            errorBounceI -= signum(EnergyFlowReal[i]*errorI*abs(error[i])*weightsActive[j][i]);

                            errorJ = signum(EnergyFlowReal[i]*errorBounceJ*weightsActive[j][i]);
                            errorBounceJ -= signum(EnergyFlowReal[j]*(errorJ*abs(error[j]))*weightsActive[i][j]);
                        }
*/

                        //float equilibrium = ((error[i])+(signum(EnergyFlowReal[j]*error[j]*weightsActive[i][j]))) + ((error[j])+(signum(EnergyFlowReal[i]*error[i]*weightsActive[j][i])));

                }else{


                    float actI = (EnergyFlowCounter[i]);
                    float actJ = (EnergyFlowCounter[j]);

                    momentum[i][j] = (lastCounter[j]*lastReal[j])*error[i]*learningRate+0.9*momentum[i][j];

                    weightsActive[i][j] += (lastCounter[j]*lastReal[j])*(error[i]+momentum[i][j]*0.1)*learningRate;

                }

                //weightsActive[j][i] += (lastReal[j]*lastCounter[j])*error[j]*error[i]*(lastReal[i]*lastCounter[i])*learningRate;
                //weightsActive[i][j] *= 0.9999;
                //weightsActive[j][i] += (lastReal[j]*lastCounter[j])*(error[i]+lastReal[i]*lastCounter[i]*error[j]*signum(weightsActive[j][i]))*learningRate;

                //weightsNeurons[j] += (lastReal[j]*lastCounter[j])*((error[i]+(lastReal[i]*lastCounter[i])*error[j]*signum(weightsNeurons[i]*weightsActive[j][i]))*learningRate);
                //weightsActive[j][i] += ((lastReal[j]*lastCounter[j]))*(error[i])*learningRate*signum(weightsNeurons[j]);


                //weightsNeurons[j] += lastCounter[j]*lastReal[j]*error[i]*learningRate*signum(weightsActive[i][j]);
                //weightsActive[j][i] += ((lastReal[j]*lastCounter[j]))*(error[i])*learningRate*signum(weightsNeurons[i]);


                //weightsNeurons[i] += lastCounter[j]*lastReal[j]*error[i]*learningRate;
                //weightsNeurons[i] += lastCounter[j]*lastReal[j]*error[i]*learningRate*signum(weightsNeurons[j]+weightsNeurons[i]);
                //weightsNeurons[i] += lastCounter[j]*error[i]*learningRate;
                //if(i >= numInputs) weightsNeurons[j] += EnergyFlowCounter[i]*error[j]*learningRate;
                //if(i >= numInputs) weightsNeurons[j] += EnergyFlowCounter[i]*error[j]*learningRate;

                //if(fireCounter[j] == 1.0 )weightsActive[j][i] += ((lastReal[j]*lastCounter[j]))*(error[i])*lastCounter[i]*learningRate;
                //else weightsInactive[j][i] += ((lastReal[j]*lastCounter[j]))*(error[i])*(1.0-lastCounter[i])*learningRate;

                //if(fireReal[j] == 1.0 )weightsActive[i][j] += ((lastReal[j]*lastCounter[j]))*(error[i])*lastReal[i]*learningRate;
                //else weightsInactive[i][j] += ((lastReal[j]*lastCounter[j]))*(error[i])*(1.0-lastReal[i])*learningRate;

                //if(fireReal[j] == fireReal[i])weightsActive[i][j] += (lastReal[j]*lastCounter[j]*(error[i]))*learningRate;
                //else weightsInactive[i][j] += (lastReal[j]*lastCounter[j]*(error[i]))*learningRate;

                //if(fireCounter[i] == 1.0)weightsActive[i][j] += (lastReal[j]*lastCounter[j]*(error[i]))*learningRate;
                //else weightsInactive[i][j] += (lastReal[j]*lastCounter[j]*(error[i]))*learningRate;

                //if(fireReal[i] == 1.0)weightsActive[i][j] += (lastReal[j]*(error[i]))*learningRate;
                //else weightsInactive[i][j] += (lastReal[j]*(error[i]))*learningRate;

                //weights[j][i] += (lastCounter[j]*error[i])*learningRate;

                //weights[j][i] += (2.0/(1.0+exp(-EnergyFlowCounter[j]*EnergyFlowReal[j]*(error[i])*weightsNeurons[j]*weightsNeurons[i]))-1.0)*learningRate;

                //weightsNeurons[j] += EnergyFlowCounter[j]*(error[i])*(2.0/(1.0-exp(-weightsNeurons[j]*weightsNeurons[i]))-1.0)*learningRate;
                //weights[j][i] += (EnergyFlowReal[j]*EnergyFlowCounter[j])*(error[i])*EnergyFlowCounter[i]*(1.0-EnergyFlowCounter[i])*learningRate;
                //weights[j][i] += (EnergyFlowReal[j]*EnergyFlowCounter[j])*(error[i])*learningRate;
                //weights[j][i] += (EnergyFlowReal[j]*EnergyFlowCounter[j])*(error[i])*(ActivityCounter[i])*(1.0-ActivityCounter[i])*learningRate;
                //weights[j][i] += (EnergyFlowReal[j]*EnergyFlowCounter[j])*(error[i])*(ActivityCounter[i])*(1.0-ActivityCounter[i])*learningRate;
                //weights[j][i] += (EnergyFlowReal[j])*(error[i])*learningRate+momentum[i][j]*0.0;
/*
                momentum[i][j] += (ActivityCounter[i])*error[j]*learningRate;
                weights[i][j] += (ActivityCounter[i])*error[j]*learningRate+momentum[i][j]*0.0;

                momentum[i][j] += (ActivityCounter[i])*error[j]*learningRate;
                weights[i][j] += (ActivityCounter[i])*error[j]*learningRate+momentum[i][j]*0.5;


                momentum[j][i] *= 0.9;

                momentum[j][i] += (ActivityCounter[j])*-error[i]*learningRate;
                weights[j][i] += (ActivityCounter[j])*-error[i]*learningRate+momentum[j][i]*0.5;
*/

                //weights[j][i] += (samplerCounter[j])*(error[i])*learningRate+momentum[i][j]*0.3;

                //weights[j][i] += (lastReal[j])*(realActivation[i]-lastReal[i])*0.01;

            }

        }

        /*
        float currentError = ((1.0-(((error[i]*0.5*error[i]*0.5)))));
        int j = weights.size()-1.0;

        momentum[i][j] *= currentError;
        momentum[i][j] *= 0.99;
        momentum[i][j] += 1.0*(error[i])*learningRate;
        weights[i][j] += 1.0*(error[i])*learningRate*5.0+momentum[i][j]*0.3;
        */


        //weights[i][i] += lastCounter[i]*(realNetActivation[i]-counterActivation[i])*learningRate;
    }


/*
    for(int i = 0; i < weightsActive.size(); i++){
        float absWeights = 0.0;
        for(int j = 0; j < weightsActive.size(); j++){
            absWeights += abs(weightsActive[j][i])+abs(weightsActive[i][j]);
        }
        for(int j = 0; j < weightsActive.size(); j++){
            weightsActive[j][i] = (weightsActive[j][i])*2.0*weightsActive.size()/absWeights;
            weightsActive[i][j] = (weightsActive[i][j])*2.0*weightsActive.size()/absWeights;
        }
    }
*/

        lastErrorSave = sqError;

}
void NeuralCluster::applyLearning(){


    float meanActivation = 0.0;
    vector<bool> alreadyDone;
    for(int i = 0; i < weightsActive.size(); i++)alreadyDone.push_back(false);
    for(int m = 0; m < weightsActive.size()-1; m++){

        int i = -1;
        bool done = false;
        while(!done){
                i = rand()%(weightsActive.size()-1);
                if(alreadyDone[i] == false){
                    alreadyDone[i] = true;
                    done = true;
                }
        }

        float meanOutput = 0.0;
        float meanInput = 0.0;
        for(int j = 0; j < weightsActive.size(); j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);

            meanOutput += activationI*(activationI*weightsActive[j][i]-(1.0-activationI)*weightsActive[i][j]);
            meanInput +=  activationJ*(activationJ*weightsActive[i][j]-(1.0-activationJ)*weightsActive[j][i]);

        }
        meanOutput /= weightsActive.size();
        meanInput /= weightsActive.size();

        for(int j = 0; j < weightsActive.size(); j++){
            float activationI = (EnergyFlowReal[i]);
            float activationJ = (EnergyFlowReal[j]);


            momentum[j][i] *= 0.99;(activationI)*(1.0-activationI);
            momentum[i][j] *= 0.99;(activationJ)*(1.0-activationJ);

            momentum[j][i] += ((activationJ*meanOutput));
            momentum[i][j] += ((activationI*meanInput));


            weightsActive[j][i] -= activationJ*abs(0.5-activationI)*(meanOutput+momentum[j][i])*0.01;
            weightsActive[i][j] -= activationI*abs(0.5-activationJ)*(meanInput+momentum[i][j])*0.01;
        }

        weightsActive[i][weightsActive.size()-1] -= (meanInput)*0.01;
    }


    float absWeights = 0.0;
    for(int i = 0; i < weightsActive.size(); i++)alreadyDone[i] = (false);
    for(int m = 0; m < weightsActive.size(); m++){

        int i = -1;
        bool done = false;
        while(!done){
                i = rand()%(weightsActive.size());
                if(alreadyDone[i] == false){
                    alreadyDone[i] = true;
                    done = true;
                }
        }

        float absWeightsOut = 0.0;
        float absWeightsIn = 0.0;
        for(int j = 0; j < weightsActive.size(); j++){
            absWeightsOut += abs(weightsActive[j][i]);
            absWeightsIn += abs(weightsActive[i][j]);
        }

        absWeights += absWeightsIn;

        for(int j = 0; j < weightsActive.size(); j++){
            weightsActive[j][i] = (weightsActive[j][i])*weightsActive.size()/absWeightsOut;
            weightsActive[i][j] = (weightsActive[i][j])*weightsActive.size()/absWeightsIn;
            if(i == j) weightsActive[i][j] = 0.0;
        }
    }




    for(int i = 0; i < weightsActive.size(); i++){
        for(int j = 0; j < weightsActive.size(); j++){
            //weightsActive[i][j] = (weightsActive[i][j])*weightsActive.size()*weightsActive.size()/absWeights;
        }
    }




    /*

    for(int i = numInputs+numOutputs; i < weightsActive.size()-1; i++){
        int j = numInputs+numOutputs+(rand()%(weightsActive.size()-1-(numInputs+numOutputs)));
        int k = numInputs+numOutputs+(rand()%(weightsActive.size()-1-(numInputs+numOutputs)));

        float activationI = (EnergyFlowReal[i]*EnergyFlowCounter[i]);
        float activationJ = (EnergyFlowReal[j]*EnergyFlowCounter[j]);
        float activationK = (EnergyFlowReal[k]*EnergyFlowCounter[k]);

        weightsActive[i][j] -= ((-activationK*weightsActive[i][j]*weightsActive[k][i]+activationJ*weightsActive[i][j]))*activationJ*0.01;
        weightsActive[k][i] -= ((-activationK*weightsActive[k][i]+activationI))*activationI*0.01;
    }
    */
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
    return (2.0/(1.0+(exp(-x))))-1.0;
}

vector<float> NeuralCluster::getActivation(){
    return EnergyFlowCounter;
}

vector<float> NeuralCluster::getTarget(){
    return EnergyFlowReal;
}

float NeuralCluster::minMax(float x){

    //float signum = 1.0;
    //if(x < 0.0) x = 0.0;
    //if(x >= 0.0) signum = 1.0;

/*
    if( x > 0.0) x = (2.0/(1.0+(exp(-x))))-1.0;
    else x = 0.0;
*/

    return (1.0/(1.0+(exp(-x))));
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


    for(int i = numInputs+numOutputs; i < numInputs+numOutputs+numHiddens+numRekurrent; i++){


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


            float weightsMean = 0.0;

            for(int j = 0; j < weightsActive[i].size(); j++){



                    InputSignalCounter +=  fireCounter[j]*(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyCounter)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireCounter[j]-fireCounter[i]));
                    InputSignalReal += fireReal[j]*(weightsActive[i][j]+deltaMatrix[i][j])*(1.0/absEnergyReal)*weightsActive.size()*weightsActive.size()*energy;//*(1.0-abs(fireReal[j]-fireReal[i]));

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
            samplerRealInputSignal[i] = InputSignalReal;

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
            EnergyFlowReal[i] =  minMax(samplerRealInputSignal[i]);

            mean[i] = (42.0*mean[i]+samplerRealInputSignal[i])/(43.0);
            meanChanging[i] = (42.0*meanChanging[i]+abs(EnergyFlowReal[i]-lastReal[i]))/(43.0);



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

        //if(hiddenWrite) for(int i = 0; i < input.size(); i++) samplerCounter[numInputs+numOutputs+numHiddens+i] = (samplerCounter[numInputs+numOutputs+numHiddens+i])*input[i];
        //if(hiddenWrite) for(int i = 0; i < input.size(); i++) samplerReal[numInputs+numOutputs+numHiddens+i] = (samplerReal[numInputs+numOutputs+numHiddens+i])*input[i];


        //samples++;
}
