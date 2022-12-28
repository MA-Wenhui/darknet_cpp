#ifndef IMAGE_H
#define IMAGE_H

#include <stdint.h>

#include <fstream>
#include <memory>
#include <string_view>
#include <opencv2/opencv.hpp>

enum ImageFormat : uint8_t {
  BGR_F32,
  BGR_U8,
  GRAY_F32,
  GRAY_U8,
};

class Image {
 public:
  Image(int w, int h, int c) : w_(w), h_(h), c_(c) {
    data_.reset(new float[w_*h_*c_]);
  }

  float& at(int x, int y, int c) {
    assert(x < w_ && y < h_ && c < c_);
    return data_[c * h_ * w_ + y * w_ + x];
  }

  float get(int x, int y, int c) {
    assert(x < w_ && y < h_ && c < c_);
    return data_[c * h_ * w_ + y * w_ + x];
  }
  void set(int x, int y, int c, float val) {
    if (x < 0 || y < 0 || c < 0 || x >= w_ || y >= h_ || c >= c_) return;
    assert(x < w_ && y < h_ && c < c_);
    data_[c * h_ * w_ + y * w_ + x] = val;
  }
  void add(int x, int y, int c, float val) {
    assert(x < w_ && y < h_ && c < c_);
    data_[c * h_ * w_ + y * w_ + x] += val;
  }
  int w_;
  int h_;
  int c_;
  std::shared_ptr<float[]> data_;
};

Image resize_image(Image im, int w, int h)
{
    Image resized(w, h, im.c_);   
    Image part(w, im.h_, im.c_);
    int r, c, k;
    float w_scale = (float)(im.w_- 1) / (w - 1);
    float h_scale = (float)(im.h_ - 1) / (h - 1);
    for(k = 0; k < im.c_; ++k){
        for(r = 0; r < im.h_; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w_ == 1){
                  val = im.get(im.w_ - 1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * im.at(ix, r, k) + dx * im.at(ix+1, r, k);
                }
                part.set(c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c_; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * part.at( c, iy, k);
                resized.at( c, r, k)=val;
            }
            if(r == h-1 || im.h_ == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * part.get( c, iy+1, k);
                resized.at(c, r, k)+=val;
            }
        }
    }

    return resized;
}


Image load_image_cv(std::string filename, int channels) {
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    cv::Mat m = cv::imread(filename, flag);
    if(m.empty()){
      throw std::runtime_error("read img failed.");
    }
    Image im(m.cols, m.rows, m.channels());
    cv::cvtColor(m, m, cv::COLOR_BGR2RGB);

    cv::Mat fimg(cv::Size(m.cols, m.rows), CV_32FC3);

    for (int k = 0; k < m.channels(); k++) {
      for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                fimg.at<cv::Vec3f>(i, j)[k] = m.at<cv::Vec3b>(i, j)[k] / 255.0;
            }
      }
    }
    memcpy(im.data_.get(), fimg.data, fimg.dataend - fimg.datastart);
    return im;
}

Image load_image(std::string filename, int w, int h, int c)
{
    Image out = load_image_cv(filename, c);

    if((h && w) && (h != out.h_ || w != out.w_)){
        Image resized = resize_image(out, w, h);
        out = resized;
    }
    return out;
}



std::shared_ptr<float[]> PrepareImage(cv::Mat im, int w, int h) {
  int new_w = im.cols;
  int new_h = im.rows;
  if (((float)w / im.cols) < ((float)h / im.rows)) {
    new_w = w;
    new_h = (im.rows * w) / im.cols;
  } else {
    new_h = h;
    new_w = (im.cols * h) / im.rows;
  }
  cv::Mat resized(cv::Size(new_w,new_h),CV_32FC3);

  cv::resize(im,resized,cv::Size(new_w,new_h));

  cv::Mat boxed(cv::Size(w, h), CV_32FC3);
  for (int i = 0; i < boxed.rows; i++) {
    for (int j = 0; j < boxed.cols; j++) {
      boxed.at<cv::Vec3f>(i, j) = {0.5,0.5,0.5};
    }
  }

  resized.copyTo(boxed(cv::Rect((boxed.cols - resized.cols) / 2,
                                (boxed.rows - resized.rows) / 2, resized.cols,
                                resized.rows)));

  SPDLOG_DEBUG("prepare boxed Image type: {}, 32FC3: {}, 8UC3: {}",
               boxed.type(), CV_32FC3, CV_8UC3);
  cv::imwrite("boxed.jpg", boxed);
  std::shared_ptr<float[]> ret(
      new float[boxed.cols * boxed.rows * boxed.channels()]);

  memcpy(ret.get(), boxed.data, (boxed.dataend - boxed.datastart));
  return ret;
}

#endif