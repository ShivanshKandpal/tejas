#include<bits/stdc++.h>
#include "tensor.h"
int main(){
    // basic test
    // Tensor t({2,3,1});
    // std::cout<<t.at({1,2,0})<<"\n";
    // t.at({1,2,0}) = 0.05f;
    // t.print();
    
    //matmul test
    Tensor a({2, 3}, {1,2,3,4,5,6});
    Tensor b({3,2}, {1,2,3,4,5,6});
    Tensor c = matmul(a,b);
    c.print();
    
    
}   