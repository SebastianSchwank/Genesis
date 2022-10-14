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
    void sleep();
    void resetStates();

private:
    vector<int> beforeLastState;
    vector<int> lastState;
    vector<int> state;

    vector<float> meanState;
    vector<float> stateChangeActivity;

    vector<float> integrator;

    vector<vector<float>> interpretationMatrix;

    vector<vector<int>> displacementMatrix;
    vector<vector<int>> similarityMatrix;

    vector<int> num_samples_on,last_num_samples_on;
    vector<int> num_samples_off,last_num_samples_off;

};

#endif // NEURALORDER_H
