#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;

vector<float> x = {1, 2, 3, 4, 5};
vector<float> y = {2, 3, 5, 6, 7};

class LinearRegression {
public:
    float m;
    float b;
    float learning_rate;

    LinearRegression(float m = 0.0, float b = 0.0, float learning_rate = 0.01){
        this->m = m;
        this->b = b;
        this->learning_rate = learning_rate;
    }

    vector<float> predict(const vector<float>& x) {
        vector<float> y_pred;
        for (float xi : x){
            y_pred.push_back(m * xi + b);
        }
        return y_pred;
    }

    float loss_function(const vector<float>& y, const vector<float>& y_pred) {
        int n = y.size();
        float sum = 0.0;
        for(int i=0; i<n; i++){
          sum += pow((y_pred[i]-y[i]), 2);
        }
        return sum/n;
    }
    
    void gradient_descent(const vector<float>& x, const vector<float>& y, const vector<float> y_pred){
        int n = y.size();
        float dm = 0.0;
        float db = 0.0;

        for(int i=0; i<n; i++){
            float error = y_pred[i]-y[i];
            dm += error * x[i];
            db += error;
        }

        dm = (-2.0/n)*dm;
        db = (-2.0/n)*db;

        m = m - learning_rate*dm;
        b = b - learning_rate*db;
    }
    float train(const vector<float>& x, const vector<float>& y, int epochs = 100){
        float loss = 0.0;
        for(int i=0; i<epochs; i++){
            vector<float> y_pred = predict(x);
            loss = loss_function(y, y_pred);
            gradient_descent(x, y, y_pred);
        }
        return loss;
    }
};


int main() {
    LinearRegression model(1, 1, 0.01);
    float final_loss = model.train(x, y, 100);

    cout << "Final Loss: " << final_loss << endl;
    cout << "Learned slope (m): " << model.m << endl;
    cout << "Learned intercept (b): " << model.b << endl;
    return 0;
}