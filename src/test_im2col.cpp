#include <iomanip>
#include <iostream>

#include "im2col.h"
#include "tensor.h"
#include "utils.h"
constexpr int H = 3;
constexpr int W = 4;
constexpr int B = 1;
constexpr int C = 3;

constexpr int PAD = 1;
constexpr int STRIDE = 2;
constexpr int SIZE = 3;

Tensor<float> random_tensor(int n, int h, int w, int c, TensorFormat format) {
  std::shared_ptr<float[]> data(new float[n * h * w * c * 2]);
  for (int i = 0; i < n * h * w * c; i++) {
    data[i] = rand_normal<float>(0, 1);
  }
  Tensor<float> tensor(n, h, w, c, format, data);
  return tensor;
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean) {
  float scale = 1. / (batch * spatial);
  int i, j, k;
  for (i = 0; i < filters; ++i) {
    mean[i] = 0;
    for (j = 0; j < batch; ++j) {
      for (k = 0; k < spatial; ++k) {
        int index = j * filters * spatial + i * spatial + k;
        mean[i] += x[index];
      }
    }
    mean[i] *= scale;
  }
}

int main() {
  auto TA = random_tensor(B, H, W, C, NHWC);
  TA.toNCHW();

  constexpr int col_im_w = (W + 2 * PAD - SIZE) / STRIDE + 1;
  constexpr int col_im_h = (H + 2 * PAD - SIZE) / STRIDE + 1;
  constexpr int M = 1;
  constexpr int K = SIZE * SIZE * C;
  constexpr int N = col_im_w * col_im_h;
  auto col_im = std::shared_ptr<float[]>(new float[K * N]);
  im2col_cpu(TA.data_.get(), C, H, W, SIZE, STRIDE, PAD, col_im.get());

  for (int i = 0; i < K; i++) {
    if (i / (SIZE * SIZE) == 0) std::cerr << TERMINAL_COLOR_RED;
    if (i / (SIZE * SIZE) == 1) std::cerr << TERMINAL_COLOR_YELLOW;
    if (i / (SIZE * SIZE) == 2) std::cerr << TERMINAL_COLOR_BLUE;
    for (int j = 0; j < N; j++) {
      std::cerr << std::setw(15) << col_im[i * N + j]
                << (j == (N - 1) ? "\n" : ",");
    }
    std::cerr << TERMINAL_COLOR_RESET;
  }

  TA.print();

  auto mean = std::shared_ptr<float[]>(new float[C]);
  mean_cpu(TA.data_.get(), B, C, H * W, mean.get());

  for (int i = 0; i < C; i++) {
    std::cout << mean[i] << "  ";
  }
  std::cout << std::endl;
}