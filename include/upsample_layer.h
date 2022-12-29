#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H

#include "common.h"
#include "layer.h"

class Network;

class UpsampleLayer : public Layer {
 public:
  UpsampleLayer(const toml::table& config,int batch, int h, int w,
                int c)
      : Layer(batch, h, w, c) {
    auto stride = get_config<int>(config, "stride"sv, 1);

    init(batch, h, w, c, stride);
  }
  ~UpsampleLayer() {}

  virtual void forward(Tensor<float>& input, Tensor<float>& output,
                       std::map<int, Tensor<float>> tensor_map) {

    output.batch_ = batch_;
    output.c_ = out_c_;
    output.h_ = out_h_;
    output.w_ = out_w_;
    output.data_size_ = batch_ * out_c_ * out_h_ * out_w_;

    for (int b = 0; b < input.batch_; b++) {
      for (int c = 0; c < input.c_; c++) {
        for (int h = 0; h < input.h_; h++) {
          for (int w = 0; w < input.w_; w++) {
            output.at(b, h * 2, w * 2, c) = input.at(b, h, w, c);
            output.at(b, h * 2 + 1, w * 2 + 1, c) = input.at(b, h, w, c);
            output.at(b, h * 2, w * 2 + 1, c) = input.at(b, h, w, c);
            output.at(b, h * 2 + 1, w * 2, c) = input.at(b, h, w, c);
          }
        }
      }
    }
  }
  virtual void backward() {}

 protected:
  void init(int batch, int h, int w, int c, int stride) {
    type_ = LAYER_TYPE::UPSAMPLE;
    stride_ = stride;
    batch_ = batch;
    w_ = w;
    h_ = h;
    c_ = c;
    out_w_ = stride_ * w_;
    out_h_ = stride_ * h_;
    out_c_ = c_;

    output_length_ = out_w_ * out_h_ * out_c_;
    input_length_ = w_ * h_ * c_;
    delta_ = new float[batch_ * output_length_];
  }

 public:
  int stride_;
  float* delta_;
};

#endif
