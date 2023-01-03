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

    bool running = false;

    int numLessons = 1;//sizeof (input)/sizeof (input[0]);
    int iteration = 0;
    float currentErrorBP,CurrentErrorMine;
    float lastErrorBP,lastErrorMine;

    int lengthOfRndVec = 128;
    vector<float> generatedRandomVector;
    int currentFrequency = 0;
    float phase = 0;
    int numInputs = 32;//sizeof (input[0])/sizeof (input[0][0]);
    int numOutputs = 3;//sizeof (output[0])/sizeof (output[0][0]);
    int numHiddens = 32;
    int numAttentions = 0;
    NeuralCluster* Cluster0;
    NeuralCluster* Cluster1;
    QTimer *timer;
    QGraphicsScene* ErrorView;

};

#endif // MAINWINDOW_H
