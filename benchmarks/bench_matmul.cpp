#include<iostream>
#include "tensor.h"
#include<chrono>
int main(){
    std::cout<<"size  |"<<"  naive  |"<<"  tiled  |"<<"  speedup  \n";
    for(int sz = 64;sz<=1024;sz*=2){
        std::vector<float> a_data,b_data;
        for(int i = 0;i<sz*sz;i++){a_data.push_back(rand()%10); b_data.push_back(rand()%10);};
        Tensor a({sz,sz}, a_data);
        Tensor b({sz,sz}, b_data);
        auto startnaive = std::chrono::high_resolution_clock::now();
        Tensor c = matmul(a, b);
        auto endnaive = std::chrono::high_resolution_clock::now();
        double msnaive = std::chrono::duration<double, std::milli>(endnaive-startnaive).count();
        
        auto starttiled = std::chrono::high_resolution_clock::now();
        Tensor d = matmul_tiled(a, b);
        auto endtiled = std::chrono::high_resolution_clock::now();
        
        double mstiled = std::chrono::duration<double,std::milli>(endtiled - starttiled).count();
        std::cout<<sz<<"  "<<msnaive<<"  "<<mstiled<<"  "<<msnaive/mstiled<<"  \n";
    }
}