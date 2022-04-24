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
    void inputData(vector<float> input,vector<float> output,bool inverted);
    void propergate(vector<float> input, vector<float> output, bool sleep, bool hiddenWrite, bool againstEmpty);
    vector<vector<float>> getWeights();
    void train(float learningRate);
    void removeNonlin(float learningRate);
    void trainBP(vector<float> target,float learningRate,int iterations);
    vector<float> getTarget();
    float signum(float x);
    float minMax(float x);
    void resetSampler(bool randomize);
    void envolve();

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
    vector<float>         derived;

    vector<float>         realActivity;
    vector<float>         counterActivity;

    vector<float>         samplerReal;
    vector<float>         samplerCounter;


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
    vector<vector<float>> weightsInactive;
    vector<float>         weightsNeurons;
    vector<vector<float>> momentum;
    vector<vector<int>>   firingMatrixCounter;
    vector<float>         neuronalActivity;
    vector<float>         backpropError;
    float                 overallError;

    vector<float>         momentumVector;
    int                   numInputs,numOutputs,numHiddens,numRekurrent;

    vector<float>         derivedError;
    vector<float>         error,lastError,beforeLasteError;

    float                 samples;

};

#endif // NEURALCLUSTER_H
