#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<float> x = {1, 2, 3, 4, 5};
vector<float> y = {0, 0, 1, 1, 1};

class LogisticRegression{
    public:
    float m;
    float b;
    float learning_rate;

    LogisticRegression(float m = 0.0, float b = 0.0, float learning_rate = 0.01){
        this->m = m;
        this->b = b;
        this->learning_rate = learning_rate;
    }

    float sigmoid(float z) {
        return 1.0f / (1.0f + exp(-z));
    }

    float predict(float x) {
        float linear = m * x + b;
        return sigmoid(linear);
    }

    float loss_function(const vector<float>& y, const vector<float>& y_pred){
        int n = y.size();
        float binary_cross_entropy = 0.0;
        for(int i=0; i<n; i++){
            binary_cross_entropy += (y[i]*log(y_pred[i] + 1e-9))+((1-y[i])*(log(1-y_pred[i] + 1e-9)));
        }
        return (-1.0f/n)*binary_cross_entropy;
    }

    void gradient_descent(const vector<float>& x, const vector<float>& y, const vector<float>& y_pred){
        int n = y.size();
        float dm = 0.0;
        float db = 0.0;
        
        for(int i=0; i<n; i++){
            float error = y_pred[i] - y[i];
            dm += error * x[i];
            db += error;
        }

        dm = (1.0f/n)*dm;
        db = (1.0f/n)*db;

        m = m - learning_rate*dm;
        b = b - learning_rate*db;
    }

    float train(const vector<float>& x, const vector<float>& y, int epochs = 100){
        float loss = 0.0;
        for(int i=0; i<epochs; i++){
            vector<float> y_pred;
            for (int j = 0; j < x.size(); ++j) {
                y_pred.push_back(predict(x[j]));
            }
            loss = loss_function(y, y_pred);
            gradient_descent(x, y, y_pred);
        }
        return loss;
    }
};


int main() {
    LogisticRegression model(0.5, -0.5, 0.1);
    float final_loss = model.train(x, y, 1000);

    cout << "Final Loss: " << final_loss << endl;
    cout << "Learned slope (m): " << model.m << endl;
    cout << "Learned intercept (b): " << model.b << endl;

    return 0;
}