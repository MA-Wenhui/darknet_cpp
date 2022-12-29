#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "gemm.h"
#include "im2col.h"
#include "layer.h"

class Network;

class ConvolutionalLayer : public Layer {
 public:
  ConvolutionalLayer(const toml::table& config, int batch, int h, int w, int c,
                     int adam)
      : Layer(batch, h, w, c) {
    int n = get_config<int>(config, "filters", 1);
    int size = get_config<int>(config, "size", 1);
    int stride = get_config<int>(config, "stride", 1);
    int pad = get_config<int>(config, "pad", 0);
    int padding = pad ? size / 2 : 0;
    int groups = get_config<int>(config, "groups", 1);
    ACTIVATION activation = string2Activation(
        get_config<std::string_view>(config, "activation", "logistic"));
    int batch_normalize = get_config<int>(config, "batch_normalize", 0);

    init(n, groups, size, stride, padding, activation, batch_normalize, adam);
  }
  ~ConvolutionalLayer() {}

  void normalize_cpu(float* x, float* mean, float* variance, int batch,
                     int filters, int spatial) {
    int b, f, i;
    for (b = 0; b < batch; ++b) {
      for (f = 0; f < filters; ++f) {
        for (i = 0; i < spatial; ++i) {
          int index = b * filters * spatial + f * spatial + i;
          x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
        }
      }
    }
  }

  virtual void forward(Tensor<float>& input, Tensor<float>& output,
                       std::map<int, Tensor<float>> tensor_map) override {
    // printf("input\n");
    // for (int i = 0; i < input.h_; i++) {
    //   for (int j = 0; j < input.w_; j++) {
    //     printf("%f ", input.data_.get()[i * input.w_ + j]);
    //   }
    //   printf("\n");
    // }
    // printf("weights\n");
    // for (int i = 0; i < filter_num_; i++) {
    //   for (int j = 0; j < c_ * size_ * size_; j++) {
    //     printf("%f ", weights_[i * c_ * size_ * size_ + j]);
    //   }
    //   printf("\n");
    // }
    int M = filter_num_;         // 卷积核数目
    int K = size_ * size_ * c_;  // 卷积核展开为一维的长度，
    int N = out_w_ * out_h_;     // 卷积核在图片上滚动的位置
    //
    auto output_ptr = output.data_.get();
    for (int batch_i = 0; batch_i < batch_; batch_i++) {
      float* a = weights_.get();  // MxK 的卷积核一维展开
      std::shared_ptr<float[]> b(new float[K * N]);
      memset(b.get(), 0, K * N * sizeof(float));
      if (size_ == 1) {
        // b = input.data_;
        memcpy(b.get(), input.data_.get(), K * N * sizeof(float));
      } else {
        im2col_cpu(input.data_.get(), c_, h_, w_, size_, stride_, pad_,
                   b.get());
      }
      // float sumb = 0, suma = 0, sumc = 0;
      // for (int j = 0; j < M * K; j++) {
      //   suma += a[j];
      // }
      // printf("a!!!    %f\n", suma);
      // for (int j = 0; j < K * N; j++) {
      //   sumb += b[j];
      // }
      // printf("b!!!    %f\n", sumb);
      // for (int j = 0; j < M * N; j++) {
      //   sumc += output_ptr[j];
      // }
      // printf("c!!!    %f\n", sumc);

      // gemm_cpu(M, N, K, 1.0, a, K, b.get(), N, output_ptr, N);
      gemm(0, 0, M, N, K, 1, a, K, b.get(), N, 1, output_ptr, N);
      // sumb = 0, suma = 0, sumc = 0;
      // for (int j = 0; j < M * K; j++) {
      //   suma += a[j];
      // }
      // printf("a--    %f\n", suma);
      // for (int j = 0; j < K * N; j++) {
      //   sumb += b[j];
      // }
      // printf("b--    %f\n", sumb);
      // for (int j = 0; j < M * N; j++) {
      //   sumc += output_ptr[j];
      // }
      // printf("c--    %f\n", sumc);
    }

    // printf("rolling_mean\n");
    // for (int i = 0; i < out_c_; i++) {
    //   printf("%f ", rolling_mean_[i]);
    // }
    // printf("\n");
    // printf("rolling_variance\n");
    // for (int i = 0; i < out_c_; i++) {
    //   printf("%f ", rolling_variance_[i]);
    // }
    // printf("\n");
    // bach normalize
    if (batch_normalize_) {
      // do normalize

      // normalize_cpu(output_ptr, rolling_mean_, rolling_variance_, batch_,
      //               out_c_, out_h_ * out_w_);

      for (int n = 0; n < batch_; n++) {
        for (int c = 0; c < out_c_; c++) {
          auto m = rolling_mean_[c];
          auto v = rolling_variance_[c];
          auto s = scales_[c];
          auto b = biases_[c];
          for (int i = 0; i < out_h_ * out_w_; i++) {
            int index = n * out_c_ * out_h_ * out_w_ + c * out_h_ * out_w_ + i;
            output_ptr[index] = (output_ptr[index] - m) / (sqrt(v) + .000001f);
            output_ptr[index] *= s;
            output_ptr[index] += b;
          }
        }
      }
    } else {
      // add bias only
      for (int n = 0; n < batch_; n++) {
        for (int c = 0; c < out_c_; c++) {
          auto b = biases_[c];
          for (int i = 0; i < out_h_ * out_w_; i++) {
            int index = n * out_c_ * out_h_ * out_w_ + c * out_h_ * out_w_ + i;
            output_ptr[index] += b;
          }
        }
      }
    }

    // activate
    activate_array(output_ptr, output_length_ * batch_, activation_);

    output.batch_ = batch_;
    output.c_ = out_c_;
    output.h_ = out_h_;
    output.w_ = out_w_;
    output.data_size_ = batch_ * out_c_ * out_h_ * out_w_;
  }

  virtual void backward() override {}
  void load_weights(std::fstream& fs) override {
    fs.read((char*)biases_, bias_num_ * sizeof(float));
    if (batch_normalize_) {
      fs.read((char*)scales_, sizeof(float) * filter_num_);
      fs.read((char*)rolling_mean_, sizeof(float) * filter_num_);
      fs.read((char*)rolling_variance_, sizeof(float) * filter_num_);
    }
    fs.read((char*)weights_.get(), nweights_ * sizeof(float));
    SPDLOG_TRACE("ConvolutionalLayer load_weights");
  }

 protected:
  void init(int n, int groups, int size, int stride, int padding,
            ACTIVATION activation, int batch_normalize, int adam) {
    this->type_ = CONVOLUTIONAL;
    groups_ = groups;
    filter_num_ = n;  // filter num
    stride_ = stride;
    size_ = size;
    pad_ = padding;
    batch_normalize_ = batch_normalize;

    nweights_ = c_ / groups_ * filter_num_ * size_ * size_;
    weights_.reset(new float[nweights_]);
    weights_updates_ = new float[nweights_];

    bias_num_ = filter_num_;
    biases_ = new float[filter_num_];
    bias_updates_ = new float[filter_num_];
    float scale = sqrt(2.0 / (size_ * size_ * c_ / groups_));
    for (int i = 0; i < nweights_; i++) {
      weights_[i] = scale * rand_normal<float>(0, 1);
    }

    out_h_ = get_out_height();
    out_w_ = get_out_width();
    out_c_ = filter_num_;
    output_length_ = out_h_ * out_w_ * out_c_;
    input_length_ = w_ * h_ * c_;

    // delta_.reset(new float[batch_ * output_length_]);

    workspace_size_ = get_workspace_size();

    activation_ = activation;

    if (batch_normalize_) {
      scales_ = new float[filter_num_];
      scales_updates_ = new float[filter_num_];
      for (int i = 0; i < filter_num_; i++) {
        scales_[i] = 1.0;
      }
      mean_ = new float[filter_num_];
      variance_ = new float[filter_num_];

      mean_delta_ = new float[filter_num_];
      variance_delta_ = new float[filter_num_];

      rolling_mean_ = new float[filter_num_];
      rolling_variance_ = new float[filter_num_];
    }

    gflops_ =
        (2.0 * filter_num_ * size_ * size_ * c_ / groups_ * out_w_ * out_h_) /
        1000000000.0;
  }

  uint64_t get_workspace_size() {
    return out_h_ * out_w_ * size_ * size_ * c_ / groups_ * sizeof(float);
  }
  int get_out_height() { return (w_ + 2 * pad_ - size_) / stride_ + 1; }
  int get_out_width() { return (h_ + 2 * pad_ - size_) / stride_ + 1; }

 public:
  int size_;  // conv_size
  int groups_;
  int pad_;
  int stride_;
  int filter_num_;

  int bias_num_;
  float* biases_;
  float* bias_updates_;

  std::shared_ptr<float[]> delta_;

  uint64_t workspace_size_;

  int nweights_;
  std::shared_ptr<float[]> weights_;
  float* weights_updates_;

  int batch_normalize_;
  float* scales_;
  float* scales_updates_;
  float* mean_;
  float* variance_;
  float* mean_delta_;
  float* variance_delta_;
  float* x_;
  float* x_norm_;

  float* rolling_mean_;
  float* rolling_variance_;

  float gflops_;
};

#endif
