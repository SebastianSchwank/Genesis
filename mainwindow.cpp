#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer = new QTimer(this);

    ErrorView = new QGraphicsScene;

    //Initalize completly connected stacked neural net
    Cluster0 = new NeuralCluster(numInputs,numOutputs,numHiddens,numAttentions);
    Cluster3 = new NeuralOrder(numInputs+numOutputs,numHiddens+numAttentions);

    image = new QImage(Cluster0->getActivation().size(),Cluster0->getActivation().size()*2,QImage::Format_RGB32);
    imageResp = new QImage((numInputs+numOutputs+numHiddens+numAttentions+1),numLessons*numOutputs*4,QImage::Format_RGB32);

    imageScaled = new QImage(Cluster0->getActivation().size()*2,Cluster0->getActivation().size()*4,QImage::Format_RGB32);
    imageRespScaled = new QImage((numInputs+numOutputs+numHiddens+numAttentions+1)*4,numLessons*numOutputs*4*4,QImage::Format_RGB32);

    scene2 = new QGraphicsScene;
    scene1 = new QGraphicsScene;

    lastErrorBP = 0.0;
    lastErrorMine = 0.0;
    currentErrorBP = 0.0;
    CurrentErrorMine = 0.0;

}

MainWindow::~MainWindow()
{
    delete ui;
}

vector<float> MainWindow::inputFunction(int type, int length,float frequency,float phase){
    vector<float> function;
    /*
    if(type == 0){
        int toggle = 0;
        for(int i = 0; i < length; i++){
            if(((i+phase)%(periode)) == 0) toggle = 1-toggle;
            if(toggle == 0) function.push_back(1.0);
            if(toggle == 1) function.push_back(0.0);
        }
    }
    if(type == 1){
        for(int i = 0; i < length; i++){
            if(((i+phase)%(periode)) == 0) function.push_back(1.0);
            else function.push_back(0.0);
        }
    }
    */
    if(type == 2){
        for(int i = 0; i < length; i++){
          function.push_back(0.5+(0.5*(sin(2.0*3.16*((frequency*i/length)+phase)))));
        }
    }
    if(type == 3){
        for(int i = 0; i < length; i++){
          function.push_back((1.0*rand()/RAND_MAX)*sin(2.0*3.16*((frequency*i/length)+phase)));
        }
    }
    return function;
}

void MainWindow::processNet(){
        float sumErrorOver = 0.0;
        float sumErrorOverBP = 0.0;

        //Test pass error calculation
        //currentFrequency = (currentFrequency+1)%(numOutputs-numLessons-1);
        int frequency = currentFrequency+3;//(((currentFrequency)%(numOutputs+8)));
        //phase = (rand())%(numOutputs+numLessons+24);
        //phase = 1;


        //frequency = 4;
        float lastError = 0.0;
        vector<vector<float>> impulseResonses;
        for(int o = 0; o < 1; o++){
            //frequency = o+2;//(rand()+1)%(numOutputs-numLessons-1);
            //phase = rand()%(frequency*8)+1.0;
            //frequency =  (rand()+1)%(numOutputs-numLessons-1);;

/*
            for(int k = 0; k < numLessons; k++){

                //Create empty vector as output placeholder
                    vector<float> emptyV;
                //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))

                    vector<float> inputV = MainWindow::inputFunction(k,numInputs,frequency*2,phase);

                    vector<float> targetV;
                    for(int j = 0; j < numOutputs; j++){
                        targetV.push_back(0.0);
                        //cout << inputV[j];
                    }
                    targetV[(k)] = 1.0;
                    targetV[numLessons-1+frequency] = 1.0;


                    for(int i = 0; i < 16; i++){
                        ClusterBP->propergate(inputV,emptyV,false);
                    }
                    vector<float> out1 = ClusterBP->getActivation();

                    for(int i = 0; i < numOutputs; i++){
                        sumErrorOverBP += (targetV[i]-out1[i+numInputs])*(targetV[i]-out1[i+numInputs]);
                        //out0[i+numInputs] = abs(targetV[i]-out0[i+numInputs]);
                    }

                    impulseResonses.push_back(out1);
           }

*/

            //phase += 0.07;
            //if(phase >= 1.0) phase = 0.0;

           for(int f = 0; f < (numOutputs); f++){
            //Training pass

               //float phase = (1.0*rand()/RAND_MAX);
               int k = f;//rand()%numOutputs;

               //Create empty vector as output placeholder
                   vector<int> emptyV;
                   vector<float> emptyVO;
               //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))

                   vector<float> inputV = MainWindow::inputFunction(2,numInputs,0.5*(k+2),phase);
                   vector<float> targetV;// = MainWindow::inputFunction(2,numInputs,k+2,phase);

                   for(int i = 0; i < numInputs; i++) emptyV.push_back(0.0);

                   for(int i = 0; i < numOutputs; i++) targetV.push_back(0.0);
                   targetV[k] = 1.0;

  /*
                   float max = 0.0;
                   for(int i = 0; i < numOutputs; i++) if(max < abs(targetV[i])) max = abs(targetV[i]);
                   float mean = 0.0;
                   for(int i = 0; i < numOutputs; i++) mean += (targetV[i]/max);
                   for(int i = 0; i < numOutputs; i++) targetV[i] = (targetV[i]/max)-(mean/(numOutputs));
*/
                    vector<int> inputB;
                    vector<int> inputBO;
                    for(int i = 0; i < numInputs; i++){
                        inputB.push_back((int)round(inputV[i]));
                        inputBO.push_back((int)round(inputV[i]));
                    }

                    for(int i = 0; i < numOutputs; i++){
                        inputBO.push_back((int)round(targetV[i]));
                    }

                    Cluster3->resetStates();
                    for(int i = 0; i < 8; i++){
                        Cluster3->propergate(inputB);
                        Cluster3->propergate(emptyV);
                    }
                    vector<float> out1;
                    out1 = Cluster3->propergate(inputB);
                    vector<float> out0;

                    Cluster3->resetStates();
                    for(int i = 0; i < 8; i++){
                        Cluster3->propergate(emptyV);
                        Cluster3->propergate(inputBO);
                        Cluster3->train();
                    }

                    for(int i = 0; i < out1.size(); i++){
                        out0.push_back((float)out1[i]);
                    }

                    impulseResonses.push_back(out0);

/*
                    for(int i = 0; i < numInputs; i++){
                        inputB[i] = 1-inputB[i];
                        inputBO[i] = 1-inputBO[i];
                    }

                    Cluster3->resetStates();
                    for(int i = 0; i < 8; i++){
                        Cluster3->propergate(inputB);
                        Cluster3->propergate(emptyV);
                    }

                    vector<float> out3;

                    out1 = Cluster3->propergate(inputB);

                    for(int i = 0; i < out1.size(); i++){
                        out3.push_back(((float)out1[i]));
                    }

                    impulseResonses.push_back(out3);

                    Cluster3->resetStates();
                    for(int i = 0; i < 8; i++){
                        Cluster3->propergate(emptyV);
                        Cluster3->propergate(inputBO);
                        Cluster3->train();
                    }

*/
                    Cluster3->resetStates();
                    for(int i = 0; i < 24; i++){
                        //Cluster3->sleep();
                    }





                    for(int i = 0; i < numOutputs; i++){
                        sumErrorOver += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                        //sumErrorOver += (targetV[i]-out3[i+numInputs])*(targetV[i]-out3[i+numInputs]);
                        //out0[i+numInputs] = abs(targetV[i]-out0[i+numInputs]);
                    }



                    /*
                   Cluster0->resetSampler(false);
                   for(int i = 0; i < 9; i++){
                       Cluster0->propergate(inputV,emptyVO,1.0);
                   }

                   //vector<float> out0 = Cluster0->getActivation();
                   impulseResonses.push_back(out0);

                   float squaredError = 0.0;

                   for(int i = 0; i < numOutputs; i++){
                       sumErrorOver += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                       squaredError += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                       //out0[i+numInputs] = abs(targetV[i]-out0[i+numInputs]);
                   }

                   float learningRate = (sqrt(squaredError/numOutputs))-lastError;
                    lastError = sqrt(squaredError/numOutputs);

                   Cluster0->resetSampler(false);
                   for(int i = 0; i < 9; i++){
                       Cluster0->propergate(inputV,targetV,(1.0-lastError));
                       for(int m = 0; m < 5; m++)Cluster0->applyLearning(0.05);
                       //Cluster0->train(learningRate*0.01,lastError);
                   }
                   //Cluster0->resetDeltaMatrix();

                   out0 = Cluster0->getTarget();
                   impulseResonses.push_back(out0);


                   Cluster0->resetSampler(false);
                   vector<float> invInput;
                   for(int i = 0; i < numInputs; i++) invInput.push_back(1.0-inputV[i]);
                   for(int i = 0; i < 9; i++){
                       Cluster0->propergate(invInput,targetV,(1.0-lastError));
                       for(int m = 0; m < 5; m++)Cluster0->applyLearning(0.05);
                       //Cluster0->train(learningRate*0.01,lastError);
                   }
*/


/*
                   Cluster0->resetSampler();
                   Cluster0->propergate(emptyV,emptyV,false,false,false);
                   for(int i = 0; i < 8; i++){
                       Cluster0->propergate(emptyV,emptyV,false,false,true);
                   }
                   Cluster0->removeNonlin(0.02);
*/






/*
                   Cluster0->resetSampler();
                   for(int i = 0; i < 128; i++){
                       Cluster0->propergate(inputV,targetV,false,false,true);
                   }
                   Cluster0->train(0.01);
*/


                    /*
                    vector<float> mergedTarget;

                    for(int i = 0; i < numInputs; i++){
                        mergedTarget.push_back(inputV[i]);
                    }

                    for(int i = 0; i < numOutputs; i++){
                        mergedTarget.push_back(targetV[i]);
                    }


                    for(int i = 0; i < 16; i++){
                        ClusterBP->propergate(inputV,emptyV,true);
                    }
                    ClusterBP->trainBP(mergedTarget,0.05,5);
                    */
            }

      }
        //Cluster0->resetDeltaMatrix();

        if(iteration%1 == 0){


        for(int x = 0; x < impulseResonses.size(); x++){
            for(int y = 0; y < impulseResonses[0].size(); y++){
                QColor col = QColor(128,128,128);
                float colorVal = (2.0/(1.0+(exp(-impulseResonses[x][y]))))-1.0;
                if(colorVal > 0.0) col = QColor(255.0*abs(colorVal),255.0*abs(0.0),255.0*abs(0.0));
                if(colorVal < 0.0) col = QColor(255.0*abs(0.0),255.0*abs(0.0),255.0*abs(colorVal));
                imageResp->setPixel(y,x,col.rgb());
            }
        }

/*
        float max = 0.0;
        for(int x = 0; x < Cluster0->getActivation().size(); x++){
            for(int y = 0; y < Cluster0->getActivation().size(); y++){
                if(abs(Cluster0->getWeights()[y][x]) > max) max = abs(Cluster0->getWeights()[y][x]);
            }
        }


        for(int x = 0; x < Cluster0->getActivation().size(); x++){
            for(int y = 0; y < Cluster0->getActivation().size(); y++){
                QColor col = QColor(128,128,128);

                if(Cluster0->getWeights()[y][x] > 0.0) col = QColor(255.0*(2.0/(1.0+(exp(-Cluster0->getWeights()[y][x])))-1.0),0,0);
                if(Cluster0->getWeights()[y][x] < 0.0) col = QColor(0,0,-255.0*(2.0/(1.0+(exp(-Cluster0->getWeights()[y][x])))-1.0));

                image->setPixel(x,y,col.rgb());
            }
        }
*/
        scene1->clear();
        scene2->clear();

        }


        if(!imageScaled->isNull()) delete imageScaled;
        if(!imageRespScaled->isNull()) delete imageRespScaled;
        imageScaled = new QImage(Cluster0->getActivation().size()*6,Cluster0->getActivation().size()*6,QImage::Format_RGB32);
        imageRespScaled = new QImage((numInputs+numOutputs+numHiddens+numAttentions+1)*8,numLessons*numOutputs*4*4,QImage::Format_RGB32);
        QSize pixSize1 = imageScaled->size();
        QSize pixSize2 = imageRespScaled->size();
        *imageScaled = (image->scaled(pixSize1, Qt::KeepAspectRatio));
        *imageRespScaled = (imageResp->scaled(pixSize2, Qt::KeepAspectRatio));


        //image = new QImage(ClusterBP->getActivation().size()*2,ClusterBP->getActivation().size()*2,QImage::Format_RGB32);

/*
        for(int x = 0; x < ClusterBP->getActivation().size()*2; x++){
            for(int y = 0; y < ClusterBP->getActivation().size()*2; y++){
                QColor col = QColor(128,128,128);

                if(ClusterBP->getWeights()[y/2][x/2] > 0.0) col = QColor(255.0*1.0/(1.0+exp(-ClusterBP->getWeights()[y/2][x/2])),0,0);
                if(ClusterBP->getWeights()[y/2][x/2] < 0.0) col = QColor(0,0,255.0*1.0/(1.0+exp(ClusterBP->getWeights()[y/2][x/2])));

                image->setPixel(x,y,col.rgb());
            }
        }
*/

        lastErrorMine = CurrentErrorMine;
        CurrentErrorMine = sumErrorOver;

        lastErrorBP = currentErrorBP;
        currentErrorBP = sumErrorOverBP;

        QPen coloredLine;
        QColor col = QColor(255,128,128);
        coloredLine.setColor(col);
        ErrorView->addLine(iteration,0,iteration+1,0);
        ErrorView->addLine(iteration,-lastErrorBP*4,iteration+1,-currentErrorBP*4);
        ErrorView->addLine(iteration,-lastErrorMine*2,iteration+1,-CurrentErrorMine*2,coloredLine);

           //scene1->addPixmap(QPixmap::fromImage(*imageScaled));
           scene2->addPixmap(QPixmap::fromImage(*imageRespScaled));

           ui->graphicsView_2->setScene(scene2);
           //ui->graphicsView->setScene( scene1 );


           ui->graphicsView_3->setScene(ErrorView);

           //ui->graphicsView->show();
           ui->graphicsView_2->show();
           ui->graphicsView_3->show();

           iteration += 1;

}

void MainWindow::on_pushButton_clicked()
{
    if(running == false){
        connect(timer, SIGNAL(timeout()), this, SLOT(processNet()));
        timer->start();
        running = !running;
    }else {
        timer->stop();
        running = !running;
    }
}

void MainWindow::on_pushButton_2_clicked()
{
    this->processNet();
}
