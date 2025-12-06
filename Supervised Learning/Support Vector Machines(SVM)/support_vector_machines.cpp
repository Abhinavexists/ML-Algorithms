#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class SupportVectorMachine {
public:
    float learning_rate;
    float regularization;

    SupportVectorMachine(float learning_rate = 0.01, float regularization = 0.01) {
        this->learning_rate = learning_rate;
        this->regularization = regularization;
    }

    float decision_boundary(const VectorXf& x, const VectorXf& w, float b) {
        return x.dot(w) + b;
    }

    float hinge_loss(const MatrixXf& X, const VectorXf& Y, const VectorXf& w, float b) {
        int n = X.rows();

        VectorXf decision = X * w + VectorXf::Constant(n, b);
        VectorXf loss_vec = (VectorXf::Constant(n, 1.0f) - Y.cwiseProduct(decision)).cwiseMax(0.0f);

        return loss_vec.mean();
    }

    std::pair<VectorXf,float> train(const MatrixXf& X, const VectorXf& Y, int epochs = 100) {

        int n = X.rows();
        int d = X.cols();

        VectorXf w = VectorXf::Zero(d);
        float b = 0.0f;

        for(int i = 0; i < epochs; i++) {
            for(int j = 0; j < n; j++) {

                VectorXf xi = X.row(j);
                float yi = Y(j);

                float decision = xi.dot(w) + b;
                bool condition = (yi * decision >= 1.0f);

                VectorXf dw;
                float db;

                if (condition) {
                    dw = 2.0f * regularization * w;
                    db = 0.0f;
                } else {
                    dw = 2.0f * regularization * w - yi * xi;
                    db = -yi;
                }

                w -= learning_rate * dw;
                b -= learning_rate * db;
            }
        }

        return {w, b};
    }
};

int main() {

    MatrixXf X(7,1);
    X << 1,2,3,4,5,6,7;

    VectorXf Y(7);
    Y << 1,1,1,1,-1,-1,-1;

    SupportVectorMachine svm(0.01f, 0.01f);
    auto [w, b] = svm.train(X, Y, 200);

    std::cout << "w = " << w.transpose() << std::endl;
    std::cout << "b = " << b << endl;

    float loss = svm.hinge_loss(X, Y, w, b);
    std::cout << "loss = " << loss << endl;

    return 0;
}
