// Defines the dataset schema and file reference for the CVD dataset.
// MLFeatures uses only the 6 features selected via Pearson correlation analysis
// (gender, height, smoke, alco, active were excluded due to low correlation with cardio).

EXPORT CardioData := MODULE

    EXPORT Layout := RECORD
        UNSIGNED3 id;
        UNSIGNED1 age;
        UNSIGNED1 gender;
        UNSIGNED2 height;
        UNSIGNED2 weight;
        UNSIGNED2 ap_hi;
        UNSIGNED2 ap_lo;
        UNSIGNED1 cholesterol;
        UNSIGNED1 gluc;
        UNSIGNED1 smoke;
        UNSIGNED1 alco;
        UNSIGNED1 active;
        UNSIGNED1 cardio;
    END;

    EXPORT MLFeatures := RECORD
        UNSIGNED3 id;
        UNSIGNED1 age;
        UNSIGNED2 weight;
        UNSIGNED2 ap_hi;
        UNSIGNED2 ap_lo;
        UNSIGNED1 cholesterol;
        UNSIGNED1 gluc;
        UNSIGNED1 cardio;
    END;

    EXPORT File := DATASET('~training::cardio_processed', Layout, CSV(HEADING(1), SEPARATOR(',')));
END;
