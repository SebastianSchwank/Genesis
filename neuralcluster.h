#ifndef NEURALCLUSTER_H
#define NEURALCLUSTER_H


#include <vector>
#include <iostream>
#include <cmath>
#include <QDebug>


//Ideas:
//Probailistic Firering
//Dropout
//Momentum

using namespace std;

class NeuralCluster
{
public:
    NeuralCluster(int inputs, int outputs, int hidden, int attention);
    void inputData(vector<float> input, vector<float> output);
    void propergate(vector<float> input, vector<float> output, float energy);
    vector<vector<float>> getWeights();
    void train(float learningRate, float sqError);
    void removeNonlin(float learningRate);
    void trainBP(vector<float> target,float learningRate,int iterations);
    vector<float> getTarget();
    float signum(float x);
    float minMax(float x);
    void resetSampler(bool randomize);
    void applyLearning();
    void resetDeltaMatrix();

    vector<float> getActivation();

    void syncronize();


private:

    int                   maxPeriod = 1;

    vector<float>         fireCounter;
    vector<float>         counterActivation;
    vector<float>         beforelastCounter;
    vector<float>         lastCounter;
    vector<float>         polarityCounter;
    vector<float>         fireReal;
    vector<float>         realActivation;
    vector<float>         lastReal;
    vector<float>         beforelastReal;
    vector<float>         polarityReal;
    vector<float>         longTermMean;

    vector<float>         mean;
    vector<float>         meanChanging;

    vector<float>         realActivity;
    vector<float>         counterActivity;

    vector<float>         samplerRealInputSignal,samplerRealOutputSignal;
    vector<float>         samplerCounterInputSignal,samplerCounterOutputSignal;

    vector<float>         samplerRealInput,samplerRealOutput,samplerRealEnergyBillance;
    vector<float>         samplerCounterInput,samplerCounterOutput,samplerCounterEnergyBillance;



    vector<float>         OutputSamplerReal;
    vector<float>         OutputSamplerCounter;

    vector<float>         integratorReal;
    vector<float>         integratorCounter;

    vector<float>         OutputIntegratorReal;
    vector<float>         OutputIntegratorCounter;

    vector<float>         EnergyFlowReal;
    vector<float>         EnergyFlowCounter;

    vector<float>         ActivityReal;
    vector<float>         ActivityCounter;

    vector<int>           counter;
    vector<int>           period;

    vector<float>         slowness;

    vector<float>         slope;
    vector<vector<float>> weightsActive;
    vector<vector<float>> biasesActive;
    vector<vector<float>> deltaMatrix;
    vector<float>         weightsNeurons;
    vector<vector<float>> momentum;
    vector<vector<int>>   firingMatrixCounter;
    vector<vector<float>> relativeBehaviour;

    vector<float>         momentumVector;
    int                   numInputs,numOutputs,numHiddens,numRekurrent;

    vector<float>         derivedError;
    vector<float>         error,lastError,beforeLasteError,approxError;
    vector<float>         outputError,inputError;

    float                 samples,lastErrorSave;

};

#endif // NEURALCLUSTER_H
