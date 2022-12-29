#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "common.h"
#include "layer.h"

class Network;

class YoloLayer : public Layer {
 public:
  YoloLayer(const toml::table& config, int batch, int h, int w, int c)
      : Layer(batch, h, w, c) {
    auto classes = get_config<int>(config, "classes"sv, 20);
    auto anchors = get_config_vec_int(config, "anchors"sv);
    auto mask = get_config_vec_int(config, "mask"sv);
    auto num = get_config<int>(config, "num"sv, 1);
    auto jitter = get_config<float>(config, "jitter"sv, 0.3f);
    auto ignore_thresh = get_config<float>(config, "ignore_thresh"sv, 0.5f);
    auto truth_thresh = get_config<float>(config, "truth_thresh"sv, 1);
    auto random = get_config<int>(config, "random"sv, 1);

    init(num, mask, classes, jitter, ignore_thresh, truth_thresh, random,
         anchors);
  }
  ~YoloLayer() {}

  virtual void forward(Tensor<float>& input, Tensor<float>& output,
                       std::map<int, Tensor<float>> tensor_map) {
    memcpy(output.data_.get(), input.data_.get(),
           input.data_size_ * sizeof(float));

    for (int b = 0; b < input.batch_; b++) {
      for (int n = 0; n < mask_.size(); n++) {
        //
        int offset =
            b * input.data_size_ + n * input.w_ * input.h_ * (classes_ + 5);
        // activate x y
        activate_array(output.data_.get() + offset, 2 * input.w_ * input.h_,
                       LOGISTIC);
        // activate classes and confidence score
        activate_array(output.data_.get() + offset + 4 * input.w_ * input.h_,
                       (1 + classes_) * input.w_ * input.h_, LOGISTIC);
      }
    }
    output.batch_ = batch_;
    output.c_ = out_c_;
    output.h_ = out_h_;
    output.w_ = out_w_;
    output.data_size_ = batch_ * out_c_ * out_h_ * out_w_;
  }
  virtual void backward() {}

 protected:
  void init(int num, std::vector<int> mask, int classes, float jitter,
            float ignore_th, float truth_th, int random,
            std::vector<int> anchors) {
    type_ = LAYER_TYPE::YOLO;

    classes_ = classes;
    num_ = num;
    out_h_ = h_;
    out_w_ = w_;

    /*
    3    *   ( 80     +    4  +   1)
    |           |          |      |
    boxes    classes     rect    confidence
               x,y,w,h, c, C1,C2...,Cn
    */
    out_c_ = c_; 
    biases_ = new float[num_ * 2];

    mask_ = mask;
    if (mask_.empty()) {
      for (int i = 0; i < num_; ++i) {
        mask.emplace_back(i);
      }
    }
    bias_updates_ = new float[num_ * 2];
    input_length_ = h_ * w_ * c_;
    output_length_ = input_length_;

    delta_ = new float[batch_ * output_length_];

    jitter_ = jitter;
    ignore_thresh_ = ignore_th;
    truth_thresh_ = truth_th;
    random_ = random;
    anchors_ = anchors;
  }

 public:
  int num_;  // num of boxes
  int classes_;
  float cost_;

  float* biases_;
  float* bias_updates_;

  float* delta_;
  float* output_;

  const uint64_t truths_ = 90 * (4 + 1);

  std::vector<int> mask_;
  std::vector<int> anchors_;

  float jitter_;
  float ignore_thresh_;
  float truth_thresh_;
  float random_;
};

#endif
