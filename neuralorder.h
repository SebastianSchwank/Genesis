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
    void train(vector<int> inputs);
    void resetStates();

private:
    vector<int> beforeLastState;
    vector<int> lastState;
    vector<int> state;

    vector<float> integrator;

    vector<vector<vector<int>>> orderMatrixOn;
    vector<vector<vector<int>>> orderMatrixOff;

    vector<int> num_samples_On;
    vector<int> num_samples_Off;
};

#endif // NEURALORDER_H
