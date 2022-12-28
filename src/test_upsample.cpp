#include <iomanip>
#include <iostream>

#include "im2col.h"
#include "tensor.h"
#include "utils.h"
constexpr int H = 3;
constexpr int W = 4;
constexpr int B = 1;
constexpr int C = 3;

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    if(forward) out[out_index] = scale*in[in_index];
                    else in[in_index] += scale*out[out_index];
                }
            }
        }
    }
}

Tensor<float> random_tensor(int n, int h, int w, int c, TensorFormat format) {
  std::shared_ptr<float[]> data(new float[n * h * w * c * 2]);
  for (int i = 0; i < n * h * w * c; i++) {
    data[i] = rand_normal<float>(0, 1);
  }
  Tensor<float> tensor(n, h, w, c, format, data);
  return tensor;
}

int main() {
  auto TA = random_tensor(B, H, W, C, NCHW);
  TA.print();
  auto TB = Tensor<float>(B, H*2, W*2, C, NCHW);
  upsample_cpu(TA.data_.get(),W,H,C,B,2,1,1,TB.data_.get());
  TB.print();

  auto TC = Tensor<float>(B, H * 2, W * 2, C, NCHW);
  for (int b = 0; b < TA.batch_; b++) {
    for (int c = 0; c < TA.c_; c++) {
            for (int h = 0; h < TA.h_; h++) {
                for (int w = 0; w < TA.w_; w++) {
                    TC.at(b, h * 2, w * 2, c) = TA.at(b, h, w, c);
                    TC.at(b, h * 2 + 1, w * 2 + 1, c) = TA.at(b, h, w, c);
                    TC.at(b, h * 2, w * 2 + 1, c) = TA.at(b, h, w, c);
                    TC.at(b, h * 2 + 1, w * 2, c) = TA.at(b, h, w, c);
                }
            }
    }
  }
  TC.print();
}