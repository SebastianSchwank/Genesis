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
    vector<float> threshhold;

    vector<vector<vector<float>>> interpretationMatrix;

    vector<int> num_samples_on;
    vector<int> num_samples_off;
};

#endif // NEURALORDER_H
