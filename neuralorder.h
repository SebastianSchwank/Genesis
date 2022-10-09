#ifndef NEURALORDER_H
#define NEURALORDER_H

#include <vector>
#include <iostream>
#include <cmath>
#include <QDebug>

using namespace std;

class NeuralOrder
{
public:
    NeuralOrder(int inputs, int hidden);

    vector<int> propergate(vector<int> inputs);
    void train();
    void resetStates();

private:
    vector<int> beforeLastState;
    vector<int> lastState;
    vector<int> state;

    vector<float> integrator;
    vector<float> offSet;

    vector<vector<vector<int>>> interpretationMatrix;
    vector<vector<int>> similarityMatrix;

    vector<int> state_balance;
    vector<int> num_samples_on;
    vector<int> num_samples_off;
    int numSamples;
    int globalState;

};

#endif // NEURALORDER_H
