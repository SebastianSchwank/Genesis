#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPixmap>
#include <QTimer>
#include <QDebug>
#include <QGraphicsScene>

#include "neuralcluster.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();
    void processNet();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_pushButton_4_clicked();

private:
    vector<float> inputFunction(int type, int length, float frequency, float phase);

private:
    Ui::MainWindow *ui;

    QImage *image;
    QImage *imageResp;

    QImage *imageScaled;
    QImage *imageRespScaled;

    QGraphicsScene* scene2;
    QGraphicsScene* scene1;


    int integrationSteps = 7;
    float offset = 0.0;

    int numLessons = 4;//sizeof (input)/sizeof (input[0]);
    int iteration = 0;
    float currentErrorBP,CurrentErrorMine;
    float lastErrorBP,lastErrorMine;

    int lengthOfRndVec = 128;
    vector<float> generatedRandomVector;
    int currentFrequency = 0;
    float phase = 0;
    int numInputs = 32;//sizeof (input[0])/sizeof(input[0][0]);
    int numOutputs = 4;//sizeof (output[0])/sizeof(output[0][0]);
    int numHiddens = 16;
    int numAttentions = 0;
    NeuralCluster* Cluster0;
    NeuralCluster* Cluster1;
    QTimer *timer;
    QGraphicsScene* ErrorView;

    vector<float> indexArray;

    bool train = true;
    bool running = false;
    bool graphics = true;

};

#endif // MAINWINDOW_H
