// Shuffles the dataset randomly and splits into train (48,000) and test (12,000) records.

IMPORT $;

CardioData := $.CardioData.File;
MLFeatures := $.CardioData.MLFeatures;

EXPORT CardioPrep := MODULE

    MLFeaturesExt := RECORD(MLFeatures)
        UNSIGNED rnd;
    END;

    EXPORT DataE := PROJECT(
        CardioData,
        TRANSFORM(
            MLFeaturesExt,
            SELF.rnd := RANDOM(),
            SELF := LEFT
        )
    );

    SHARED DataES := SORT(DataE, rnd);

    EXPORT TrainData := PROJECT(DataES[1..48000], MLFeatures);
    EXPORT TestData  := PROJECT(DataES[48001..60000], MLFeatures);

END;
