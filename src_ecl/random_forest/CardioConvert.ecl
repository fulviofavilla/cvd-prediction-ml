// Converts preprocessed train/test datasets into ML_Core NumericField format
// and separates independent (X) and dependent (Y) variables for model training.

IMPORT $, ML_Core;

TrainData := $.CardioPrep.TrainData;
TestData  := $.CardioPrep.TestData;

ML_Core.ToField(TrainData, TrainDataNF);
ML_Core.ToField(TestData, TestDataNF);

EXPORT CardioConvert := MODULE

    EXPORT IndTrainDataNF := TrainDataNF(number < 12);

    EXPORT DepTrainDataNF := PROJECT(
        TrainDataNF(number = 12),
        TRANSFORM(
            ML_Core.Types.DiscreteField,
            SELF.number := 1,
            SELF := LEFT
        )
    );

    EXPORT IndTestDataNF := TestDataNF(number < 12);

    EXPORT DepTestDataNF := PROJECT(
        TestDataNF(number = 12),
        TRANSFORM(
            ML_Core.Types.DiscreteField,
            SELF.number := 1,
            SELF := LEFT
        )
    );
END;
