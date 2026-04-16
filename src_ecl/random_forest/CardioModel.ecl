// Trains a Classification Forest model and evaluates it via confusion matrix,
// feature importance, and accuracy assessment.

IMPORT $, ML_Core;
IMPORT LearningTrees AS LT;

UNSIGNED numTrees             := 100;
UNSIGNED featuresPerNode      := 2;
UNSIGNED maxDepth             := 100;
SET OF UNSIGNED nominalFields := [2, 8, 9, 10, 11, 12];

Learner     := LT.ClassificationForest(numTrees, featuresPerNode, maxDepth, nominalFields);
Model       := Learner.GetModel($.CardioConvert.IndTrainDataNF, $.CardioConvert.DepTrainDataNF);
ModelStats  := Learner.GetModelStats(Model);
FImportance := Learner.FeatureImportance(Model);

PredictedDeps   := Learner.Classify(Model, $.CardioConvert.IndTestDataNF);
ConfusionMatrix := ML_Core.Analysis.Classification.ConfusionMatrix(PredictedDeps, $.CardioConvert.DepTestDataNF);
Assessment      := ML_Core.Analysis.Classification.Accuracy(PredictedDeps, $.CardioConvert.DepTestDataNF);

OUTPUT(ModelStats, NAMED('ModelStats'));
OUTPUT(PredictedDeps, NAMED('ClassifiedDeps'));
OUTPUT(ConfusionMatrix, NAMED('ConfusionMatrix'));
OUTPUT($.CardioConvert.DepTestDataNF, NAMED('CorrectOutput'));
OUTPUT(FImportance, NAMED('FeatureImportance'));
OUTPUT(Assessment, NAMED('Assessment'));
