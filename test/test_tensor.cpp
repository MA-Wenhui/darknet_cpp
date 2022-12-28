#include <gtest/gtest.h>

#include "tensor.h"
#include "utils.h"

constexpr int H = 300;
constexpr int W = 400;
constexpr int B = 32;
constexpr int C = 3;

Tensor<float> random_tensor(int n, int h, int w, int c, TensorFormat format) {
  std::shared_ptr<float[]> data(new float[n * h * w * c]);
  for (int i = 0; i < n * h * w * c; i++) {
    data[i] = rand_normal<float>(0, 1);
  }
  Tensor<float> tensor(n, h, w, c, format, data);
  return tensor;
}

TEST(TestTensor, format_transform) {
  auto TA = random_tensor(B, H, W, C, NHWC);
  EXPECT_EQ(TA.format_, NHWC);
  auto TB = TA.clone();
  TB.toNCHW();
  EXPECT_EQ(TB.format_, NCHW);
  auto TC = TB.clone();
  TC.toNHWC();
  EXPECT_EQ(TC.format_, NHWC);
  for (int i = 0; i < TA.data_size_; i++) {
    EXPECT_EQ(TA.data_[i], TC.data_[i]);
  }

  //   TA.print();
  for (int b = 0; b < B; b++) {
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        for (int c = 0; c < C; c++) {
          EXPECT_EQ(TA.at(b, i, j, c), TB.at(b, i, j, c));
          EXPECT_EQ(TA.at(b, i, j, c), TC.at(b, i, j, c));
        }
      }
    }
  }
}