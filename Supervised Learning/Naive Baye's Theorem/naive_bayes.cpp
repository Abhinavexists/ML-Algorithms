#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class NaiveBayes {
    unordered_map<float, float> class_priors;
    unordered_map<float, VectorXf> mean;
    unordered_map<float, VectorXf> variance;
    vector<float> classes;

public:
    NaiveBayes() {}

    void decision_fit(const MatrixXf& x, const VectorXf& y) {
        int n_samples  = x.rows();
        int n_features = x.cols();

        unordered_set<float> s;
        for (int i = 0; i < y.size(); i++) s.insert(y(i));
        classes.assign(s.begin(), s.end());

        for (float c : classes) {
            vector<int> idx;
            for (int i = 0; i < y.size(); i++)
                if (y(i) == c) idx.push_back(i);

            MatrixXf x_class(idx.size(), n_features);
            for (int i = 0; i < idx.size(); i++)
                x_class.row(i) = x.row(idx[i]);

            class_priors[c] = float(idx.size()) / n_samples;

            mean[c] = x_class.colwise().mean();

            VectorXf var(n_features);
            for (int j = 0; j < n_features; j++) {
                float m = mean[c](j);
                float acc = 0;
                for (int i = 0; i < x_class.rows(); i++)
                    acc += std::pow(x_class(i, j) - m, 2);
                var(j) = acc / x_class.rows();
            }
            variance[c] = var;
        }
    }

    float gaussian_probability(float x, float mean_val, float var) {
        float numerator = std::exp(-std::pow(x - mean_val, 2) / (2 * var));
        float denominator = std::sqrt(2 * M_PI * var);
        return numerator / denominator;
    }

    float predict_sample(const VectorXf& x) {
        float best_output = -1;
        float max_proba = -1;

        for (float c : classes) {
            float proba = std::log(class_priors[c]);

            for (int i = 0; i < x.size(); i++) {
                float mean_val = mean[c](i);
                float var = variance[c](i);
                proba += std::log(gaussian_probability(x(i), mean_val, var));
            }

            if (proba > max_proba) {
                max_proba = proba;
                best_output = c;
            }
        }

        return best_output;
    }

    VectorXf predict(const MatrixXf& x) {
        VectorXf predictions(x.rows());
        for (int i = 0; i < x.rows(); i++) {
            predictions(i) = predict_sample(x.row(i));
        }
        return predictions;
    }
};

int main() {
    MatrixXf X(10, 2);
    VectorXf y(10);

    X << 1.0, 2.0,
         2.0, 3.0,
         3.0, 1.0,
         4.0, 5.0,
         5.0, 6.0,
         1.5, 2.5,
         2.5, 3.5,
         3.5, 1.5,
         4.5, 5.5,
         5.5, 6.5;

    y << 0, 0, 0, 1, 1, 0, 0, 0, 1, 1;

    NaiveBayes nb;
    nb.decision_fit(X, y);

    VectorXf predictions = nb.predict(X);

    cout << "Predictions: " << predictions.transpose() << endl;

    return 0;
}
