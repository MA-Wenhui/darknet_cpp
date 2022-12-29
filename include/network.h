#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "convolutional_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "upsample_layer.h"
#include "yolo_layer.h"
#include "box.h"

#define SECRET_NUM -1234

typedef enum { MULT, ADD, SUB, DIV } BINARY_ACTIVATION;

class Network;
class Layer;
class Data;

typedef struct detection {
  box bbox;
  int classes;
  std::vector<float>prob;
  std::vector<float>mask;
  float objectness;
  int sort_class;
} detection;

typedef struct {
  int id;
  float x, y, w, h;
  float left, right, top, bottom;
} box_label;

typedef enum {
  CONSTANT,
  STEP,
  EXP,
  POLY,
  STEPS,
  SIG,
  RANDOM
} LearningRatePolicy;

LearningRatePolicy to_policy(std::string_view s) {
  if (s == "random") return RANDOM;
  if (s == "poly") return POLY;
  if (s == "constant") return CONSTANT;
  if (s == "step") return STEP;
  if (s == "exp") return EXP;
  if (s == "sigmoid") return SIG;
  if (s == "steps") return STEPS;
  std::cerr << "Couldn't find policy " << s << ", going with constant."
            << std::endl;
  return CONSTANT;
}

class Network {
 public:
  Network(std::string_view cfg_file, std::string_view weight_file, bool clear) {
    toml::table config = toml::parse_file(cfg_file);
    parse_net_config(config);
    construct_layers(config);
    load_weights(weight_file);
  }

  ~Network(){};

  void SetBatch(int batch) {
    this->batch_ = batch;
    for (auto l : layers_) {
      l->batch_ = batch_;
    }
  }

  void Forward(Tensor<float> X) {
    std::shared_ptr<float[]> buffer1(new float[max_output_length_]);
    memset(buffer1.get(), 0, max_output_length_ * sizeof(float));
    std::shared_ptr<float[]> buffer2(new float[max_output_length_]);
    memset(buffer2.get(), 0, max_output_length_ * sizeof(float));
    memcpy(buffer1.get(), X.data_.get(),
           X.data_size_ * sizeof(decltype(X)::dtype));

    Tensor<float> a(X.batch_, X.h_, X.w_, X.c_, X.format_, buffer1);
    Tensor<float> b(X.batch_, X.h_, X.w_, X.c_, X.format_, buffer2);
    for (auto l : layers_) {
      l->forward(a, b, tensor_map_);
      if(tensor_map_.find(l->index_)!=tensor_map_.end()){
        tensor_map_.at(l->index_) = b;
      }
      printf("layer%d: ", l->index_);
      for (int i = 0; i < 0+10; i++) {
        printf("%f ", b.data_.get()[i]);
      }
      printf("\n");
      // SPDLOG_DEBUG(
      //     "forward {:<4} {:>15}: {:>4} x {:>4} x {:>4} -> {:>4} x {:>4} x "
      //     "{:>4}",
      //     l->index_, l->type_str_, l->h_, l->w_, l->c_, l->out_h_, l->out_w_,
      //     l->out_c_);
      a.swap(b);
      memset(b.data_.get(),0,b.data_size_*sizeof(float));
    }
    //82 94 106
    for (auto id : {82, 94, 106}) {
      auto yolo_tensor = tensor_map_.at(id);
      auto yolo_layer = std::dynamic_pointer_cast<YoloLayer>(layers_.at(id));
      for (int i = 0; i < yolo_layer->w_ * yolo_layer->h_; i++) {
        for (int n = 0; n < yolo_layer->mask_.size(); n++) {
          float *data =
              yolo_tensor.data_.get() + n * (yolo_layer->classes_ + 5) * yolo_layer->w_ * yolo_layer->h_ + i;
          float obj_score = *(data + 4);
          if (obj_score > 0.8) {
            detection det;
            det.bbox.x = *(data + 0);
            det.bbox.y = *(data + 1);
            det.bbox.w = *(data + 2);
            det.bbox.h = *(data + 3);

            det.objectness = obj_score;
            for (int c = 0; c < yolo_layer->classes_; c++) {
              det.prob.push_back(*(data + 5 + c));
            }
            detections_.push_back(det);
          }
        }
      }
    }

  }

  void parse_net_config(const toml::table &config) {
    toml::table net_config = *(config["net"].as_table());
    // std::cout<<net_config["layers"]<<std::endl;
    this->batch_ = get_config<int>(net_config, "batch"sv, 1);
    this->learning_rate_ =
        get_config<float>(net_config, "learning_rate"sv, 0.001f);
    this->momentum_ = get_config<float>(net_config, "momentum"sv, 0.9f);
    this->decay_ = get_config<float>(net_config, "decay"sv, 0.001f);
    this->subdivisions_ = get_config<int>(net_config, "subdivisions"sv, 1);
    this->time_steps_ = get_config<int>(net_config, "time_steps"sv, 1);
    this->notruth_ = get_config<int>(net_config, "notruth"sv, 0);
    this->batch_ /= this->subdivisions_;
    this->batch_ *= this->time_steps_;
    this->random_ = get_config<int>(net_config, "random"sv, 0);
    this->adam_ = get_config<int>(net_config, "adam"sv, 0);
    if (this->adam_) {
      this->B1_ = get_config<float>(net_config, "B1"sv, 0.9f);
      this->B2_ = get_config<float>(net_config, "B2"sv, 0.999f);
      this->eps_ = get_config<float>(net_config, "eps"sv, 0.0000001f);
    }

    this->h_ = get_config<int>(net_config, "height"sv, 0);
    this->w_ = get_config<int>(net_config, "width"sv, 0);
    this->c_ = get_config<int>(net_config, "channels"sv, 0);

    this->inputs_ =
        get_config<int>(net_config, "inputs"sv, this->h_ * this->w_ * this->c_);
    this->max_crop_ = get_config<int>(net_config, "max_crop"sv, this->w_ * 2);
    this->min_crop_ = get_config<int>(net_config, "min_crop"sv, this->w_);
    this->max_ratio_ =
        get_config<float>(net_config, "max_ratio"sv,
                          static_cast<float>(this->max_crop_ / this->w_));
    this->min_ratio_ =
        get_config<float>(net_config, "min_ratio"sv,
                          static_cast<float>(this->min_crop_ / this->w_));
    this->center_ = get_config<int>(net_config, "center"sv, 0);
    this->clip_ = get_config<float>(net_config, "clip"sv, 0.0f);

    this->angle_ = get_config<float>(net_config, "angle"sv, 0.0f);
    this->aspect_ = get_config<float>(net_config, "aspect"sv, 1.0f);
    this->exposure_ = get_config<float>(net_config, "exposure"sv, 1.0f);
    this->saturation_ = get_config<float>(net_config, "saturation"sv, 1.0f);
    this->hue_ = get_config<float>(net_config, "hue"sv, 0.0f);

    if (this->inputs_ == 0 && (h_ == 0 || w_ == 0 || c_ == 0)) {
      throw std::runtime_error("\n\n\n   No input params supplied \n\n\n");
    }

    this->policy_ =
        to_policy(get_config<std::string_view>(net_config, "policy"sv, ""));
    this->burn_in_ = get_config<int>(net_config, "burn_in"sv, 0);
    this->power_ = get_config<float>(net_config, "power"sv, 4.0f);
    switch (this->policy_) {
      case LearningRatePolicy::STEP: {
        this->step_ = get_config<int>(net_config, "step"sv, 1);
        this->scale_ = get_config<float>(net_config, "scale"sv, 1.0f);
        break;
      }
      case LearningRatePolicy::STEPS: {
        this->steps_ = get_config_vec_int(net_config, "steps"sv);
        this->scales_ = get_config_vec_float(net_config, "scales"sv);
        this->num_steps_ = this->steps_.size();
        break;
      }
      case LearningRatePolicy::EXP: {
        this->gamma_ = get_config<float>(net_config, "gama"sv, 1.0f);
        break;
      }
      case LearningRatePolicy::SIG: {
        this->gamma_ = get_config<float>(net_config, "gama"sv, 1.0f);
        this->step_ = get_config<int>(net_config, "step"sv, 1);
        break;
      }
      default:
        break;
    }
    this->max_batches_ = get_config<int>(net_config, "max_batches"sv, 0);
  }

  void construct_layers(const toml::table &config) {
    auto layers_config = config["layers"];
    auto layers_arr = layers_config.as_array();
    int idx = 0;
    if (layers_arr) {
      this->n_ = layers_arr->size();
      int temp_h = this->h_;
      int temp_w = this->w_;
      int temp_c = this->c_;
      max_output_length_ = 0;
      layers_arr->for_each([&](const toml::table &ele) {
        auto lstr = get_config<std::string>(ele, "type"sv);
        auto lt = string2LayerType(lstr);
        // std::cout <<idx <<": "<<  ele["type"].value_or("") << std::endl;

        switch (lt) {
          case LAYER_TYPE::CONVOLUTIONAL: {
            SPDLOG_DEBUG("making layer: {}-{}", idx, lstr);
            auto layer_ptr = std::make_shared<ConvolutionalLayer>(
                ele, batch_, temp_h, temp_w, temp_c, this->adam_);
            SPDLOG_INFO(
                "[{}] conv {} {}x{}/{} [{} x {} x {}] -> [{} x {} x {}] {} {} "
                "GFLOPS",
                idx, layer_ptr->filter_num_, layer_ptr->size_, layer_ptr->size_,
                layer_ptr->stride_, layer_ptr->w_, layer_ptr->h_, layer_ptr->c_,
                layer_ptr->out_w_, layer_ptr->out_h_, layer_ptr->out_c_,
                layer_ptr->output_length_, layer_ptr->gflops_);
            this->layers_.push_back(layer_ptr);
            temp_h = layer_ptr->out_h_;
            temp_w = layer_ptr->out_w_;
            temp_c = layer_ptr->out_c_;

            break;
          }
          case LAYER_TYPE::YOLO: {
            SPDLOG_DEBUG("making layer: {}-{}", idx, lstr);
            auto layer_ptr = std::make_shared<YoloLayer>(ele, batch_, temp_h,
                                                         temp_w, temp_c);
            this->layers_.push_back(layer_ptr);
            SPDLOG_INFO("[{}] yolo", idx);
            temp_h = layer_ptr->out_h_;
            temp_w = layer_ptr->out_w_;
            temp_c = layer_ptr->out_c_;

            tensor_map_.insert(
                {idx, Tensor<float>(layer_ptr->batch_, layer_ptr->h_,
                                    layer_ptr->w_, layer_ptr->c_, NCHW)});
            break;
          }
          case LAYER_TYPE::ROUTE: {
            SPDLOG_DEBUG("making layer: {}-{}", idx, lstr);
            auto layer_ptr = std::make_shared<RouteLayer>(
                ele, batch_, temp_h, temp_w, temp_c, idx, this->layers_);
            this->layers_.push_back(layer_ptr);
            SPDLOG_INFO("[{}] route: {} ", idx,
                        join<int>(layer_ptr->layers_in_, ","));
            temp_h = layer_ptr->out_h_;
            temp_w = layer_ptr->out_w_;
            temp_c = layer_ptr->out_c_;

            for (auto from_idx : layer_ptr->layers_in_) {
              auto layer_from = layers_.at(from_idx);
              tensor_map_.insert(
                  {from_idx,
                   Tensor<float>(layer_from->batch_, layer_from->h_,
                                 layer_from->w_, layer_from->c_, NCHW)});
            }
            break;
          }
          case LAYER_TYPE::UPSAMPLE: {
            SPDLOG_DEBUG("making layer: {}-{}", idx, lstr);
            auto layer_ptr = std::make_shared<UpsampleLayer>(
                ele, batch_, temp_h, temp_w, temp_c);
            this->layers_.push_back(layer_ptr);
            SPDLOG_INFO("[{}] upsample {}  [{} x {} x {}] -> [{} x {} x {}]",
                        idx, layer_ptr->stride_, layer_ptr->w_, layer_ptr->h_,
                        layer_ptr->c_, layer_ptr->out_w_, layer_ptr->out_h_,
                        layer_ptr->out_c_);
            temp_h = layer_ptr->out_h_;
            temp_w = layer_ptr->out_w_;
            temp_c = layer_ptr->out_c_;
            break;
          }
          case LAYER_TYPE::SHORTCUT: {
            SPDLOG_DEBUG("making layer: {}-{}", idx, lstr);
            auto layer_ptr = std::make_shared<ShortcutLayer>(
                ele, batch_, temp_h, temp_w, temp_c, idx, this->layers_);
            this->layers_.push_back(layer_ptr);
            SPDLOG_INFO("[{}] res {}  [{} x {} x {}] -> [{} x {} x {}]", idx,
                        layer_ptr->from_index_, layer_ptr->w_, layer_ptr->h_,
                        layer_ptr->c_, layer_ptr->out_w_, layer_ptr->out_h_,
                        layer_ptr->out_c_);
            temp_h = layer_ptr->out_h_;
            temp_w = layer_ptr->out_w_;
            temp_c = layer_ptr->out_c_;
            auto layer_from = layers_.at(layer_ptr->from_index_);
            tensor_map_.insert(
                {layer_ptr->from_index_,
                 Tensor<float>(layer_from->batch_, layer_from->h_,
                               layer_from->w_, layer_from->c_, NCHW)});
            break;
          }
          default: {
            SPDLOG_CRITICAL("Unknown layer type: {}", idx);
            throw std::runtime_error("not implemented");
            break;
          }
        }
        layers_.back()->type_str_ = lstr;
        layers_.back()->index_ = idx;

        if (layers_.back()->output_length_ > max_output_length_) {
          max_output_length_ = layers_.back()->output_length_;
        }
        idx++;
      });
      SPDLOG_DEBUG("max output_length: {}", max_output_length_);
    }
  }

  void load_weights(std::string_view weight_file) {
    auto file_size = std::filesystem::file_size(weight_file);
    SPDLOG_INFO("weight file: {}", weight_file);
    SPDLOG_INFO("total size: {}", file_size);
    std::fstream fs(weight_file, std::ios::binary | std::ios::in);
    if (fs.is_open()) {
      int32_t major;
      int32_t minor;
      int32_t revision;
      fs.read((char *)&major, sizeof(int32_t));
      fs.read((char *)&minor, sizeof(int32_t));
      fs.read((char *)&revision, sizeof(int32_t));
      SPDLOG_INFO("major: {}, minor: {}, revision: {}", major, minor, revision);
      fs.read((char *)&seen_, sizeof(size_t));

      SPDLOG_TRACE("loaded: {}/{}", fs.tellg(), file_size);

      for (auto l : layers_) {
        l->load_weights(fs);
        SPDLOG_TRACE("loaded: {} / {}, {}%", fs.tellg(), file_size,
                     fs.tellg() * 100.0 / file_size);
      }
      fs.close();
    } else {
      SPDLOG_CRITICAL("weight_file: {} open failed.", weight_file);
    }
  }

 public:
  int n_;  // number of layers
  int batch_;
  size_t *seen_;
  int *t_;
  float epoch_;
  int subdivisions_;
  std::vector<LayerPtr> layers_;
  float *output_;
  uint64_t max_output_length_;

  std::map<int, Tensor<float> > tensor_map_;

  std::vector<detection> detections_;
  LearningRatePolicy policy_;

  float learning_rate_;
  float momentum_;
  float decay_;
  float gamma_;
  float scale_;
  float power_;
  int time_steps_;
  int step_;
  int max_batches_;
  std::vector<float> scales_;
  std::vector<int> steps_;
  int num_steps_;
  int burn_in_;

  int adam_;
  float B1_;
  float B2_;
  float eps_;

  int inputs_;
  int outputs_;
  int truths_;
  int notruth_;
  int h_, w_, c_;
  int max_crop_;
  int min_crop_;
  float max_ratio_;
  float min_ratio_;
  int center_;
  float angle_;
  float aspect_;
  float exposure_;
  float saturation_;
  float hue_;
  int random_;

  float *truth_;
  float *delta_;
  float *workspace_;
  int train_;
  int index_;
  float *cost_;
  float clip_;
};

#endif