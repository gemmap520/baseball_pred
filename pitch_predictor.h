#ifndef PITCH_PREDICTOR_H
#define PITCH_PREDICTOR_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double release_speed;
    int balls;
    int strikes;
    int outs;
    int inning;
    int pitch_type;  // Convert pitch types to integers (e.g., FF -> 1, SL -> 2)
} PitchData;

typedef struct {
    double *coefficients;
    int num_features;
} LogisticRegressionModel;

int load_data(const char *filename, PitchData **data, int *num_samples);
void initialize_model(LogisticRegressionModel *model, int num_features);
void train_model(LogisticRegressionModel *model, PitchData *data, int num_samples);
double predict_pitch(const LogisticRegressionModel *model, const PitchData *data);
void free_data(PitchData *data, int num_samples);

#endif
