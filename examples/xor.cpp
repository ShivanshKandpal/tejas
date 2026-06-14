#include "tensor.h"
#include<iostream> 

int main(){
    float X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    float Y[4] = {0,1,1,0};
    TensorPtr W1 = std::make_shared<Tensor>(std::vector<int> {2, 4});
    W1->randomize();
    W1->requires_grad = true;
    TensorPtr b1 = std::make_shared<Tensor>(std::vector<int>{1,4});
    b1->randomize();
    b1->requires_grad = true;
    TensorPtr W2 = std::make_shared<Tensor>(std::vector<int> {4, 1});
    W2->randomize();
    W2->requires_grad = true;
    TensorPtr b2 = std::make_shared<Tensor>(std::vector<int>{1,1});
    b2->randomize();
    b2->requires_grad = true;

    float lr = 0.1f;
    int epochs = 1000;

    for(int epoch = 0;epoch<epochs;epoch++){
        float total_loss = 0;
        for(int i = 0;i<4;i++){
            W1->zero_grad();W2->zero_grad();b1->zero_grad();b2->zero_grad();
            TensorPtr input = std::make_shared<Tensor>(std::vector<int> {1,2});
            input->data[0] = X[i][0];
            input->data[1] = X[i][1];
            TensorPtr h = relu(add(matmul(input, W1),b1));
            TensorPtr out = add(matmul(h, W2), b2);
            TensorPtr loss = mse_loss(out, Y[i]);
            loss->backward();
            sgd_step(W1, lr);
            sgd_step(W2, lr);
            sgd_step(b1, lr);
            sgd_step(b2, lr);
            total_loss += loss->data[0];
        }
        if(epoch % 100 == 0)
            std::cout << "epoch " << epoch << " loss: " << total_loss << "\n";
    }
    for(int i = 0;i<4;i++){
        TensorPtr input = std::make_shared<Tensor>(std::vector<int> {1,2});
        input->data[0] = X[i][0];
        input->data[1] = X[i][1];
        TensorPtr h = relu(add(matmul(input, W1),b1));
        TensorPtr out = add(matmul(h, W2), b2);
        TensorPtr loss = mse_loss(out, Y[i]);
        std::cout<<out->data[0]<<" "<<Y[i]<<' '<<loss->data[0]<<"\n";
    }

}