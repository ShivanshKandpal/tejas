#include<iostream>
#include "tensor.h"
#include<chrono>
#include<memory>
int main(){
    std::cout<<"size  |"<<"  naive  |"<<"  tiled  |"<<"  speedup  \n";
    for(int sz = 64;sz<=1024;sz*=2){
        std::vector<float> a_data,b_data;
        for(int i = 0;i<sz*sz;i++){a_data.push_back(rand()%10); b_data.push_back(rand()%10);};
        TensorPtr a = std::make_shared<Tensor>(std::vector<int> {sz,sz});
        a->data = a_data;
        TensorPtr b = std::make_shared<Tensor>(std::vector<int> {sz,sz});
        b->data = b_data;
        auto startnaive = std::chrono::high_resolution_clock::now();
        TensorPtr c = matmul_raw(a, b);
        auto endnaive = std::chrono::high_resolution_clock::now();
        double msnaive = std::chrono::duration<double, std::milli>(endnaive-startnaive).count();
        
        auto starttiled = std::chrono::high_resolution_clock::now();
        TensorPtr d = matmul_tiled_raw(a, b);
        auto endtiled = std::chrono::high_resolution_clock::now();
        
        double mstiled = std::chrono::duration<double,std::milli>(endtiled - starttiled).count();
        std::cout<<sz<<"  "<<msnaive<<"  "<<mstiled<<"  "<<msnaive/mstiled<<"  \n";
    }
}