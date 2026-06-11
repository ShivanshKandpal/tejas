#include<bits/stdc++.h>
#include "tensor.h"
int main(){
    Tensor t({2,3,1});
    std::cout<<t.at({1,2,0})<<"\n";
    t.at({1,2,0}) = 0.05f;
    t.print();

}