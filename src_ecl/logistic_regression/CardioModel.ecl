// Trains a Binomial Logistic Regression model and evaluates it via confusion matrix and AIC.

IMPORT LogisticRegression AS LR;
IMPORT $, ML_Core;

UNSIGNED max_iter := 260;

XTrain := $.CardioConvert.IndTrainDataNF;
YTrain := $.CardioConvert.DepTrainDataNF;
XTest  := $.CardioConvert.IndTestDataNF;
YTest  := $.CardioConvert.DepTestDataNF;

Learner := LR.BinomialLogisticRegression(max_iter,,);
Model   := Learner.getModel(XTrain, YTrain);
Predict := Learner.Classify(Model, XTest);

ConfMatrix := ML_Core.Analysis.Classification.ConfusionMatrix(Predict, YTest);
ConfAccy   := LR.BinomialConfusion(ConfMatrix);

Beta     := LR.ExtractBeta(Model);
Scores   := LR.LogitScore(Beta, XTest);
Deviance := LR.Deviance_Detail(YTest, Scores);
AIC      := LR.Model_Deviance(Deviance, Beta);

OUTPUT(Predict, NAMED('PredictedValues'));
OUTPUT(ConfMatrix, NAMED('ConfusionMatrix'));
OUTPUT(ConfAccy, NAMED('ConfusionAccuracy'));
OUTPUT(AIC, NAMED('AIC'));
