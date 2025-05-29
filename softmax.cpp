#include "softmax.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

vector<float> softmax(const vector<float>& input){
    float max = *max_element(input.begin(),input.end());
    float sum = 0.0f;
    vector<float> probs(input.size(),0.0f);
    for(int i = 0;i<10;i++){
        probs[i] = exp(input[i] - max);
        sum += probs[i];
    }
    for(float& val : probs)
        val /= sum;
    
    return probs;
}