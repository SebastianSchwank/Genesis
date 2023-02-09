#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer = new QTimer(this);

    ErrorView = new QGraphicsScene();

    //Initalize completly connected stacked neural net
    Cluster0 = new NeuralCluster(numInputs,numOutputs,numHiddens,numAttentions);

    image = new QImage(Cluster0->getActivation().size(),Cluster0->getActivation().size()*2,QImage::Format_RGB32);
    imageResp = new QImage((numInputs+numOutputs+numHiddens+numAttentions+1),numLessons*numOutputs*4*integrationSteps,QImage::Format_RGB32);

    imageScaled = new QImage(Cluster0->getActivation().size()*2,Cluster0->getActivation().size()*4,QImage::Format_RGB32);
    imageRespScaled = new QImage((numInputs+numOutputs+numHiddens+numAttentions+1)*4,numLessons*numOutputs*4*4*integrationSteps,QImage::Format_RGB32);

    scene2 = new QGraphicsScene();
    scene1 = new QGraphicsScene();

    lastErrorBP = 0.0;
    lastErrorMine = 0.0;
    currentErrorBP = 0.0;
    CurrentErrorMine = 0.0;

    for(int i = 0; i < numOutputs; i++) indexArray.push_back(rand()%numLessons);

    for(int i = 0; i < lengthOfRndVec; i++) generatedRandomVector.push_back( ((-1.0*rand()/(RAND_MAX)*abs(1.0-0.0)+1.0*rand()/RAND_MAX)*abs(0.0-1.0)/(abs(1.0-0.0)+abs(0.0-1.0))));

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
        vector<vector<vector<float>>> impulseResonses;
        offset += 0.07;//1.0*rand()/RAND_MAX;

        float squaredError = 0.0;

           for(int k = 0; k < (numLessons); k++){
            //Training pass

               vector<vector<float>> thisLesson;
                Cluster0->resetSampler(false);
                phase = 0.0;
               for(float t = 0; phase < 4.0; t++){

                   phase += 2.0/integrationSteps;

               //float phase = (1.0*rand()/RAND_MAX);
               //int k = rand()%numOutputs;


               //if(phase >= 1.0) phase = 0.0;

               //Create empty vector as output placeholder
                   vector<float> emptyV;
                   vector<float> emptyVO;
               //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))

                   vector<float> inputV = MainWindow::inputFunction(2,numInputs,2.0*((k))+1,phase+offset);
                   vector<float> targetV;// = MainWindow::inputFunction(2,numInputs,k+2,phase);

                   for(int i = 0; i < numInputs; i++) emptyV.push_back(0.0);

                   for(int i = 0; i < numOutputs; i++) targetV.push_back(0.0);
                   targetV[k] = 1.0;


                   //Cluster0->propergateImpulse(8,inputV,targetV),


                   //Cluster0->resetSampler(false);
                       for(int i = 0; i < 4; i++) Cluster0->propergate(inputV,emptyVO,0.0);


                   vector<float> out0 = Cluster0->getActivation();
                   thisLesson.push_back(out0);


                   for(int i = 0; i < numOutputs; i++){
                       sumErrorOver += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                       squaredError += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                       //out0[i+numInputs] = abs(targetV[i]-out0[i+numInputs]);
                   }


                   float learningRate = (sqrt(squaredError/numOutputs))-lastError;
                    lastError = (squaredError/numOutputs);
                    //Cluster0->propergateImpulse(9,inputV,targetV,0.5);
               }

               impulseResonses.push_back(thisLesson);
            }

           if(train){
           for(int k = 0; k < (numLessons); k++){
               phase = 0.0;
               Cluster0->resetSampler(false);

               for(float t = 0; phase < 4.0; t++){

                   phase += 2.0/integrationSteps;


               //Create empty vector as output placeholder
                   vector<float> emptyV;
                   vector<float> emptyVO;
               //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))

                   vector<float> inputV = MainWindow::inputFunction(2,numInputs,2.0*((k))+1,phase+offset);
                   vector<float> targetV;// = MainWindow::inputFunction(2,numInputs,k+2,phase);

                   for(int i = 0; i < numInputs; i++) emptyV.push_back(0.0);

                   for(int i = 0; i < numOutputs; i++) targetV.push_back(0.0);
                   targetV[k] = 1.0;

                   //Cluster0->resetSampler(false);
                   float lastSquaredErr = 1.0;
                   float sqaredErr = 0.0;
                       for(int i = 0; i < 4; i++){
                           Cluster0->propergate(inputV,emptyVO,(pow(sqaredErr/(numOutputs),0.5)));

                           vector<float> out0 = Cluster0->getActivation();
                           sqaredErr = 0.0;
                           for(int i = 0; i < numOutputs; i++){
                               sqaredErr += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                               //out0[i+numInputs] = abs(targetV[i]-out0[i+numInputs]);
                           }


                           lastSquaredErr = sqaredErr;
                           //lastSquaredErr = sqaredErr;
                           Cluster0->applyLearning(0.5,(pow(squaredError/(numOutputs*numOutputs*integrationSteps*2.0),0.5))*(1.0-pow(lastSquaredErr/(numOutputs),0.5))*exp((pow(squaredError/(numOutputs*numOutputs*integrationSteps*2.0),0.5)-pow(lastSquaredErr/(numOutputs),0.5))),k);


                           //Cluster0->applyLearning(0.125,(1.0-pow(sqaredErr/(numOutputs),0.5)),k);

                           //Cluster0->propergate(inputV,emptyVO,(1.0-lastError));
                           //Cluster0->applyLearning(0.1,squaredError/(numOutputs*numOutputs),k);
                           //Cluster0->propergate(inputV,emptyVO,(1.0-lastError));
                           //Cluster0->applyLearning(0.1,1.0,k);
                       }

/*
                       float maxOut = 0.0;
                       int maxIndex = 0;
                       for(int i = 0; i < numOutputs; i++){
                           if(out0[i+numInputs] > maxOut){
                               maxIndex = i;
                               maxOut = out0[i+numInputs];
                           }
                       }
                       int swap = indexArray[k];
                       for(int i = 0; i < numOutputs; i++) if(indexArray[i] == maxIndex) indexArray[i] = swap;
                       indexArray[k] = maxIndex;
*/
                   //Cluster0->resetDeltaMatrix();

                }
            }


           for(int k = 0; k < (numLessons); k++){
               phase = 0.0;
               Cluster0->resetSampler(false);

               for(float t = 0; phase < 4.0; t++){

                   phase += 2.0/integrationSteps;


               //Create empty vector as output placeholder
                   vector<float> emptyV;
                   vector<float> emptyVO;
               //Create input vector for holding the input data (Frequency is random Waveform depends on the lesson number (is mapped to output-neurons))

                   vector<float> inputV = MainWindow::inputFunction(2,numInputs,2.0*((k))+1,phase+offset);
                   vector<float> targetV;// = MainWindow::inputFunction(2,numInputs,k+2,phase);

                   for(int i = 0; i < numInputs; i++) emptyV.push_back(0.0);

                   for(int i = 0; i < numOutputs; i++) targetV.push_back(1.0);
                   targetV[k] = 0.0;

                   //Cluster0->resetSampler(false);
                   float lastSquaredErr = 1.0;
                   float sqaredErr = 0.0;
                       for(int i = 0; i < 4; i++){
                           Cluster0->propergate(inputV,emptyVO,(1.0-pow(sqaredErr/(numOutputs),0.5)));

                           vector<float> out0 = Cluster0->getActivation();
                           sqaredErr = 0.0;
                           for(int i = 0; i < numOutputs; i++){
                               sqaredErr += (targetV[i]-out0[i+numInputs])*(targetV[i]-out0[i+numInputs]);
                               //out0[i+numInputs] = abs(targetV[i]-out0[i+numInputs]);
                           }


                           lastSquaredErr = sqaredErr;
                           //lastSquaredErr = sqaredErr;
                           Cluster0->applyLearning(0.1,((pow(squaredError/(numOutputs*numOutputs*integrationSteps*2.0),0.5)))*(pow(lastSquaredErr/(numOutputs),0.5))*exp(-(pow(squaredError/(numOutputs*numOutputs*integrationSteps*2.0),0.5)-pow(lastSquaredErr/(numOutputs),0.5))),k);


                           //Cluster0->applyLearning(0.125,(1.0-pow(sqaredErr/(numOutputs),0.5)),k);

                           //Cluster0->propergate(inputV,emptyVO,(1.0-lastError));
                           //Cluster0->applyLearning(0.1,squaredError/(numOutputs*numOutputs),k);
                           //Cluster0->propergate(inputV,emptyVO,(1.0-lastError));
                           //Cluster0->applyLearning(0.1,1.0,k);
                       }

/*
                       float maxOut = 0.0;
                       int maxIndex = 0;
                       for(int i = 0; i < numOutputs; i++){
                           if(out0[i+numInputs] > maxOut){
                               maxIndex = i;
                               maxOut = out0[i+numInputs];
                           }
                       }
                       int swap = indexArray[k];
                       for(int i = 0; i < numOutputs; i++) if(indexArray[i] == maxIndex) indexArray[i] = swap;
                       indexArray[k] = maxIndex;
*/
                   //Cluster0->resetDeltaMatrix();

                }
            }

           }


        Cluster0->propergateEmpty(8);


        if(graphics){


            for(int less = 0; less < impulseResonses.size(); less++){
        for(int x = 0; x < impulseResonses[0].size(); x++){
            for(int y = 0; y < impulseResonses[0][0].size(); y++){
                QColor col;
                float colorVal = (2.0/(1.0+(exp(-impulseResonses[less][x][y]))))-1.0;
                col = QColor(colorVal*64,colorVal*64,colorVal*255);
                if(y < numInputs)col = QColor(colorVal*255,colorVal*64,colorVal*64);
                if((y >= numInputs) && (y < numInputs+numOutputs))col = QColor(colorVal*64,colorVal*255,colorVal*64);
                imageResp->setPixel(y,x+less*impulseResonses[0].size(),col.rgb());
            }
        }
            }


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

        scene1->clear();
        scene2->clear();




        if(!imageScaled->isNull()) delete imageScaled;
        if(!imageRespScaled->isNull()) delete imageRespScaled;
        imageScaled = new QImage(Cluster0->getActivation().size()*6,Cluster0->getActivation().size()*6,QImage::Format_RGB32);
        imageRespScaled = new QImage((numInputs+numOutputs+numHiddens+numAttentions+1)*8,numLessons*numOutputs*4*4*integrationSteps,QImage::Format_RGB32);
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

        //float movingMeanRandomnessFactor =
/*
        float unshapedMeanOver = 0.0;
        //Vary size of random Vec on Precicion needed
        for(int i = 0; i < generatedRandomVector.size(); i++) unshapedMeanOver += generatedRandomVector[i];
        unshapedMeanOver /= generatedRandomVector.size();

        float expShapedWheightedMeanOver = 0.0;

        for(int i = 0; i < generatedRandomVector.size(); i++) expShapedWheightedMeanOver += generatedRandomVector[i]*exp(0.01*i);
        expShapedWheightedMeanOver /= generatedRandomVector.size();

        if(generatedRandomVector.size() > lengthOfRndVec) generatedRandomVector.erase(generatedRandomVector.begin());


        float rndNbrProb = 10*(0.1+abs(expShapedWheightedMeanOver))*(2.0/(1.0+exp(-(-(1.0*rand()/RAND_MAX)+(1.0*rand()/RAND_MAX))))-1.0);
        CurrentErrorMine = (CurrentErrorMine*320.0+1.0*(rndNbrProb))/321.0;
        currentErrorBP = rndNbrProb;

        generatedRandomVector.push_back(rndNbrProb);
*/


        scene1->addPixmap(QPixmap::fromImage(*imageScaled));
        scene2->addPixmap(QPixmap::fromImage(*imageRespScaled));

        ui->graphicsView_2->setScene(scene2);
        ui->graphicsView->setScene( scene1 );

        }

        lastErrorMine = CurrentErrorMine;
        CurrentErrorMine = sumErrorOver/numLessons;

        lastErrorBP = currentErrorBP;
        currentErrorBP = sumErrorOverBP;

        QPen coloredLine;
        QColor col = QColor(255,128,128);
        coloredLine.setColor(col);
        ErrorView->addLine(iteration,0,iteration+1,0);
        ErrorView->addLine(iteration,-lastErrorBP*5,iteration+1,-currentErrorBP*5);
        ErrorView->addLine(iteration,-lastErrorMine*4,iteration+1,-CurrentErrorMine*4,coloredLine);



           ui->graphicsView_3->setScene(ErrorView);

           ui->graphicsView->show();
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

void MainWindow::on_pushButton_3_clicked()
{
    train = !train;
}


void MainWindow::on_pushButton_4_clicked()
{
    graphics = !graphics;
}

