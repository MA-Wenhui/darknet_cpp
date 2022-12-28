#ifndef SHORTCUT_LAYER_H
#define SHORTCUT_LAYER_H

#include "common.h"
#include "layer.h"

class Network;

class ShortcutLayer : public Layer {
 public:
  ShortcutLayer(const toml::table& config, int batch, int h, int w, int c,
                int index, const std::vector<LayerPtr>& layers)
      : Layer(batch, h, w, c) {
    int from = get_config<int>(config, "from"sv);
    int from_index = from < 0 ? (from + index) : from;
    ACTIVATION activation = string2Activation(
        get_config<std::string_view>(config, "activation", "linear"));
    auto layer_ptr = layers.at(from_index);

    init(batch, h, w, c, from_index, activation, layer_ptr->out_w_,
         layer_ptr->out_h_, layer_ptr->out_c_);
  }
  ~ShortcutLayer() {}

  virtual void forward(Tensor<float>& input, Tensor<float>& output,
                       std::map<int, Tensor<float>> tensor_map) {
    auto output_ptr = output.data_.get();
    auto from = tensor_map.at(from_index_);

    SPDLOG_DEBUG("ShortcutLayer::forward import {}", from_index_);
    printf("    layer%d: ", from_index_);
    for (int t = 0; t < 10; t++) {
      printf("%f ", from.data_.get()[t]);
    }
    printf("\n");
    assert(input.batch_ == from.batch_&& input.c_ == from.c_&& input.h_ ==
               from.h_&& input.w_ == from.w_);
    for (int i = 0; i < input.data_size_; i++) {
      output.data_[i] = input.data_[i] + from.data_[i];
    }

    // activate
    activate_array(output_ptr, output_length_ * batch_, activation_);

    output.batch_ = input.batch_;
    output.c_ = input.c_;
    output.w_ = input.w_;
    output.h_ = input.h_;
    output.data_size_ = input.data_size_;
  }
  virtual void backward() {}

 protected:
  void init(int batch, int h, int w, int c, int from, ACTIVATION activation,
            int fromw, int fromh, int fromc) {
    this->type_ = LAYER_TYPE::SHORTCUT;
    this->batch_ = batch;
    this->w_ = fromw;
    this->h_ = fromh;
    this->c_ = fromc;
    this->out_w_ = w;
    this->out_h_ = h;
    this->out_c_ = c;
    this->output_length_ = w * h * c;
    this->input_length_ = output_length_;

    this->from_index_ = from;
    this->activation_ = activation;

  }

 public:
  int from_index_;

};

#endif
