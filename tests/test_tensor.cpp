#include<bits/stdc++.h>
#include "tensor.h"

bool approx_equal(float a, float b, float eps = 1e-4f){
    return std::abs(a - b) < eps;
}

void test_tensor_basics(){
    std::cout << "=== tensor basics ===\n";
    TensorPtr t = std::make_shared<Tensor>(std::vector<int>{2, 3});
    t->at({0,0}) = 1.0f; t->at({0,1}) = 2.0f; t->at({0,2}) = 3.0f;
    t->at({1,0}) = 4.0f; t->at({1,1}) = 5.0f; t->at({1,2}) = 6.0f;
    t->print();
    std::cout << "strides: [" << t->strides[0] << ", " << t->strides[1] << "]\n";
    std::cout << (t->at({1,2}) == 6.0f ? "PASS" : "FAIL") << " at() read\n";
    t->at({1,2}) = 99.0f;
    std::cout << (t->at({1,2}) == 99.0f ? "PASS" : "FAIL") << " at() write\n\n";
}

void test_matmul_correctness(){
    std::cout << "=== matmul correctness ===\n";
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{1,2,3,4,5,6});
    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{3,2}, std::vector<float>{1,2,3,4,5,6});
    TensorPtr c_naive = matmul(a, b);
    TensorPtr c_tiled = matmul_tiled(a, b);
    std::vector<float> expected = {22,28,49,64};
    bool naive_ok = true, tiled_ok = true;
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++){
            if(!approx_equal(c_naive->at({i,j}), expected[i*2+j])) naive_ok = false;
            if(!approx_equal(c_tiled->at({i,j}), expected[i*2+j])) tiled_ok = false;
        }
    std::cout << (naive_ok ? "PASS" : "FAIL") << " naive matmul\n";
    std::cout << (tiled_ok ? "PASS" : "FAIL") << " tiled matmul\n\n";
}

void test_matmul_non_multiple_of_tile(){
    std::cout << "=== non-tile-multiple dimensions ===\n";
    int M=5, K=7, N=3;
    std::vector<float> a_data, b_data;
    for(int i=0;i<M*K;i++) a_data.push_back((float)(i+1));
    for(int i=0;i<K*N;i++) b_data.push_back((float)(i+1));
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{M,K}, a_data);
    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{K,N}, b_data);
    TensorPtr c_naive = matmul(a, b);
    TensorPtr c_tiled = matmul_tiled(a, b);
    bool ok = true;
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            if(!approx_equal(c_naive->at({i,j}), c_tiled->at({i,j}))) ok = false;
    std::cout << (ok ? "PASS" : "FAIL") << " tiled matches naive for non-tile-multiple dims\n\n";
}

void test_matmul_large(){
    std::cout << "=== large matrix naive vs tiled ===\n";
    int M=64, K=64, N=64;
    std::vector<float> a_data, b_data;
    for(int i=0;i<M*K;i++) a_data.push_back((float)(rand()%10));
    for(int i=0;i<K*N;i++) b_data.push_back((float)(rand()%10));
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{M,K}, a_data);
    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{K,N}, b_data);
    TensorPtr c_naive = matmul(a, b);
    TensorPtr c_tiled = matmul_tiled(a, b);
    bool ok = true;
    for(int i=0;i<M;i++)
        for(int j=0;j<N;j++)
            if(!approx_equal(c_naive->at({i,j}), c_tiled->at({i,j}))) ok = false;
    std::cout << (ok ? "PASS" : "FAIL") << " 64x64 tiled matches naive\n\n";
}

void test_elementwise(){
    std::cout << "=== elementwise ops ===\n";
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{1,2,3,4,5,6});
    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{6,5,4,3,2,1});
    TensorPtr c = add(a, b);
    bool add_ok = true;
    for(int i = 0; i < c->numel(); i++)
        if(!approx_equal(c->data[i], 7.0f)) add_ok = false;
    std::cout << (add_ok ? "PASS" : "FAIL") << " add\n";
    TensorPtr d = multiply(a, b);
    std::vector<float> expected_mul = {6,10,12,12,10,6};
    bool mul_ok = true;
    for(int i = 0; i < d->numel(); i++)
        if(!approx_equal(d->data[i], expected_mul[i])) mul_ok = false;
    std::cout << (mul_ok ? "PASS" : "FAIL") << " multiply\n";
    TensorPtr e = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{-3,-1,0,1,2,3});
    TensorPtr f = relu(e);
    std::vector<float> expected_relu = {0,0,0,1,2,3};
    bool relu_ok = true;
    for(int i = 0; i < f->numel(); i++)
        if(!approx_equal(f->data[i], expected_relu[i])) relu_ok = false;
    std::cout << (relu_ok ? "PASS" : "FAIL") << " relu\n\n";
}
void test_autograd(){
    std::cout << "=== autograd ===\n";
    // let's take a simple case: loss = sum(matmul(a, b))
    // a = [[1,2],[3,4]], b = [[1,0],[0,1]] (identity)
    // c = a @ b = a
    // loss = sum(c) = 1+2+3+4 = 10
    // dL/dA should be all ones (2x2)
    // dL/dB should be a^T

    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,2}, std::vector<float>{1,2,3,4});
    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{2,2}, std::vector<float>{1,0,0,1});
    a->requires_grad = true;
    b->requires_grad = true;

    TensorPtr c = matmul(a, b);

    c->grad = std::make_shared<Tensor>(c->shape);
    for(int i = 0; i < c->numel(); i++) c->grad->data[i] = 1.0f;

    if(c->backward_fn) c->backward_fn();


    std::cout << "dL/dA:\n"; a->grad->print();

    std::cout << "dL/dB:\n"; b->grad->print();

    bool da_ok = true, db_ok = true;
    for(int i = 0; i < 4; i++)
        if(!approx_equal(a->grad->data[i], 1.0f)) da_ok = false;
    std::vector<float> expected_db = {1+3, 1+3, 2+4, 2+4};
    for(int i = 0; i < 4; i++)
        if(!approx_equal(b->grad->data[i], expected_db[i])) db_ok = false;

    std::cout << (da_ok ? "PASS" : "FAIL") << " dL/dA\n";
    std::cout << (db_ok ? "PASS" : "FAIL") << " dL/dB\n\n";
}
void test_relu_backward(){
    std::cout << "=== relu backward ===\n";
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{-3,-1,0,1,2,3});
    a->requires_grad = true;

    TensorPtr b = relu(a);
    b->grad = std::make_shared<Tensor>(b->shape);
    for(int i = 0; i < b->numel(); i++) b->grad->data[i] = 1.0f;

    if(b->backward_fn) b->backward_fn();

    // derivative of relu: 1 where input > 0, else 0
    // input: [-3,-1,0,1,2,3] → grad: [0,0,0,1,1,1]
    std::vector<float> expected = {0,0,0,1,1,1};
    bool ok = true;
    for(int i = 0; i < 6; i++)
        if(!approx_equal(a->grad->data[i], expected[i])) ok = false;

    a->grad->print();
    std::cout << (ok ? "PASS" : "FAIL") << " relu backward\n\n";
}
void test_add_backward(){
    std::cout << "=== add backward ===\n";
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{1,2,3,4,5,6});
    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{6,5,4,3,2,1});
    a->requires_grad = true;
    b->requires_grad = true;

    TensorPtr c = add(a, b);
    c->grad = std::make_shared<Tensor>(c->shape);
    for(int i = 0; i < c->numel(); i++) c->grad->data[i] = 1.0f;

    if(c->backward_fn) c->backward_fn();

    // gradient of add just flows through unchanged
    // dL/dA = dL/dC = ones
    // dL/dB = dL/dC = ones
    bool a_ok = true, b_ok = true;
    for(int i = 0; i < 6; i++){
        if(!approx_equal(a->grad->data[i], 1.0f)) a_ok = false;
        if(!approx_equal(b->grad->data[i], 1.0f)) b_ok = false;
    }

    std::cout << "dL/dA: "; a->grad->print();
    std::cout << "dL/dB: "; b->grad->print();
    std::cout << (a_ok ? "PASS" : "FAIL") << " add backward dL/dA\n";
    std::cout << (b_ok ? "PASS" : "FAIL") << " add backward dL/dB\n\n";
}
void test_sum(){
    std::cout << "=== sum forward + backward ===\n";
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,3}, std::vector<float>{1,2,3,4,5,6});
    a->requires_grad = true;
    TensorPtr s = sum(a);
    // forward: 1+2+3+4+5+6 = 21
    std::cout << (approx_equal(s->data[0], 21.0f) ? "PASS" : "FAIL") << " sum forward\n";
    // backward: dL/da[i] = 1 for all i
    if(s->backward_fn){
        s->grad = std::make_shared<Tensor>(std::vector<int>{1});
        s->grad->data[0] = 1.0f;
        s->backward_fn();
    }
    bool ok = true;
    for(int i = 0; i < 6; i++)
        if(!approx_equal(a->grad->data[i], 1.0f)) ok = false;
    std::cout << (ok ? "PASS" : "FAIL") << " sum backward\n\n";
}
void test_end_to_end(){
    std::cout << "=== end to end: matmul -> relu -> sum ===\n";
    // a = [[1,2],[3,4]], b = [[1,0],[0,1]] identity
    // c = matmul(a,b) = a = [[1,2],[3,4]]
    // d = relu(c) = [[1,2],[3,4]] (all positive)
    // loss = sum(d) = 10
    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,2}, std::vector<float>{1,2,3,4});
    TensorPtr b = std::make_shared<Tensor>(std::vector<int>{2,2}, std::vector<float>{1,0,0,1});
    a->requires_grad = true;
    b->requires_grad = true;

    TensorPtr c = matmul(a, b);
    TensorPtr d = relu(c);
    TensorPtr loss = sum(d);

    std::cout << (approx_equal(loss->data[0], 10.0f) ? "PASS" : "FAIL") << " forward pass loss=10\n";

    loss->backward();

    // dL/dA should be all ones (relu passes grad through, matmul with identity passes grad through)
    bool da_ok = true;
    for(int i = 0; i < 4; i++)
        if(!approx_equal(a->grad->data[i], 1.0f)) da_ok = false;
    std::cout << (da_ok ? "PASS" : "FAIL") << " dL/dA all ones\n";
    std::cout << "dL/dA: "; a->grad->print();
    std::cout << "dL/dB: "; b->grad->print();
    std::cout << "\n";
}

void test_softmax() {
    std::cout << "=== Testing Softmax ===\n";

    TensorPtr a = std::make_shared<Tensor>(std::vector<int>{2,3});
    a->data = {
        1.0f, 2.0f, 3.0f,
        100.0f, 101.0f, 102.0f,
    };

    TensorPtr result = softmax(a);

    std::vector<float> expected = {
        0.090030f, 0.244728f, 0.665240f,
        0.090030f, 0.244728f, 0.665240f
    };

    bool match = true;
    for (int i = 0; i < result->numel(); i++) {
        float diff = std::abs(result->data[i] - expected[i]);
        if (diff > 1e-4f) { 
            std::cerr << "[FAIL] Softmax mismatch at index " << i 
                      << ": Expected " << expected[i] 
                      << ", Got " << result->data[i] << "\n";
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "[PASS] Softmax Math & Stability\n";
    } else {
        assert(false);
    }
    for(int i = 0; i < 2; i++){
        float row_sum = 0.0f;
        for(int j = 0; j < 3; j++) row_sum += result->data[i*3 + j];
        if(std::abs(row_sum - 1.0f) > 1e-5f){
            std::cerr << "[FAIL] Row " << i << " sums to " << row_sum << " not 1.0\n";
            assert(false);
        }
    }
    std::cout << "[PASS] Softmax rows sum to 1\n";

}
int main(){
    test_tensor_basics();
    test_matmul_correctness();
    test_matmul_non_multiple_of_tile();
    test_matmul_large();
    test_elementwise();
    test_autograd();
    test_relu_backward();
    test_add_backward();
    test_sum();
    test_end_to_end();
    test_softmax();
    return 0;
}