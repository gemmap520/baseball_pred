#include <stdio.h>
#include <stdlib.h>
#include "pitch_predictor.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <datafile.csv>\n", argv[0]);
        return 1;
    }

    const char *datafile = argv[1];
    PitchData *data = load_data(datafile);
    if (!data) {
        fprintf(stderr, "Error loading data\n");
        return 1;
    }

    LogisticRegressionModel model;
    initialize_model(&model);
    train_model(&model, data);

    // Example prediction
    GameSituation situation = {0, 3, 2, 1}; // 0 outs, 3 balls, 2 strikes, 1st inning
    double probability = predict_pitch(&model, &situation);
    printf("Predicted probability of fastball: %.2f%%\n", probability * 100);

    free_data(data);
    return 0;
}
