#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H

#include "common.h"
#include "layer.h"

class Network;

// route or concatenate
class RouteLayer : public Layer {
 public:
  RouteLayer(const toml::table& config,
             int batch, int h, int w, int c, int index,
             const std::vector<LayerPtr>& layers)
      : Layer(batch, h, w, c) {
    auto layers_in = get_config_vec_int(config, "layers_in"sv);

    init(batch, h, w, c, layers_in, index, layers);
  }
  ~RouteLayer() {}

  virtual void forward(Tensor<float>& input, Tensor<float>& output,
                       std::map<int, Tensor<float>> tensor_map) {
    auto output_ptr = output.data_.get();
    int offset = 0;
    for (auto i : layers_in_) {
      SPDLOG_DEBUG("RouteLayer::forward concat {}",i);

      auto in_tensor = tensor_map.at(i);
      printf("    layer%d: ", i);
      for (int t = 0; t < 10; t++) {
        printf("%f ", in_tensor.data_.get()[t]);
      }
      printf("\n");
      memcpy(output_ptr + offset, in_tensor.data_.get(),
             in_tensor.data_size_ * sizeof(decltype(in_tensor)::dtype));
      offset += in_tensor.data_size_ * sizeof(decltype(in_tensor)::dtype);
    }
    output.batch_ = batch_;
    output.c_ = out_c_;
    output.h_ = out_h_;
    output.w_ = out_w_;
  };
  virtual void backward() {}

 protected:
  void init(int batch, int h, int w, int c, std::vector<int> layers_in,
            int index, const std::vector<LayerPtr>& layers) {
    this->type_ = LAYER_TYPE::ROUTE;
    this->batch_ = batch;
    this->layers_in_ = layers_in;
    for (auto& i : layers_in_) {
      if (i < 0) {
        i += index;
      }
      this->output_length_ += layers.at(i)->output_length_;
    }
    this->input_length_ = this->output_length_;
    //尺寸不变，通道叠加
    this->out_w_ = layers.at(layers_in_.front())->out_w_;
    this->out_h_ = layers.at(layers_in_.front())->out_h_;
    this->out_c_ = layers.at(layers_in_.front())->out_c_;
    for (int i = 1; i < layers_in_.size(); i++) {
      this->out_c_ += layers.at(layers_in.at(i))->out_c_;
    }

  }

 public:
  std::vector<int> layers_in_;
};

#endif
