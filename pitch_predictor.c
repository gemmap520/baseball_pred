#include "pitch_predictor.h"

// Sigmoid function definition
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Initialize coefficients for the logistic regression model
void initialize_coefficients(double *coefficients, int num_features) {
    for (int i = 0; i < num_features; ++i) {
        coefficients[i] = 0; // Can also initialize with small random values
    }
}

// Implement the logistic regression training using gradient descent
void gradient_descent(LogisticRegressionModel *model, PitchData *data, int num_samples, double learning_rate, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int j = 0; j < model->num_features; ++j) {
            double gradient_sum = 0.0;
            for (int i = 0; i < num_samples; ++i) {
                double predicted = sigmoid(model->coefficients[0] + model->coefficients[j] * data[i].features[j]);
                gradient_sum += (predicted - data[i].pitch_type) * data[i].features[j];
            }
            model->coefficients[j] -= learning_rate * gradient_sum / num_samples;
        }
    }
}

// Use the trained logistic regression model to predict a pitch type given the features
double predict(const LogisticRegressionModel *model, const double *features) {
    double z = model->coefficients[0]; // Bias term
    for (int i = 1; i < model->num_features; ++i) {
        z += features[i] * model->coefficients[i];
    }
    return sigmoid(z);
}
