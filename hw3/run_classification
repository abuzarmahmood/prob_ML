load('HW3Data.mat')
D = NB_XGivenY(XTrain, yTrain);
p = NB_YPrior(yTrain);
yHatTrain = NB_Classify(D, p, XTrain);
yHatTest = NB_Classify(D, p, XTest);
trainError = ClassificationError(yHatTrain, yTrain);
testError = ClassificationError(yHatTest, yTest);