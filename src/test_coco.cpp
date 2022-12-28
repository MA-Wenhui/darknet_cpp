
#include <iostream>
#include <list>
#include <string_view>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "network.h"
#include "toml/toml.hpp"
#include <opencv2/opencv.hpp>
#include "image.h"

auto read_names(std::string_view data_cfg) {
  std::string ret;
  auto config = toml::parse_file(data_cfg);
  ret = std::string(config["names"].value<std::string_view>().value());
  return ret;
}

std::vector<std::string> get_coco_names(std::string name_file) {
  std::vector<std::string> ret;
  std::cout<<std::string(name_file)<<std::endl;
  std::fstream fs(name_file, std::ios::in);
  std::string name = "";
  while (fs >> name) {
    if (!name.empty()) {
      ret.push_back(name);
    }
  }
  fs.close();
  return ret;
}

int main(int argc, char **argv) {

  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%t-%H:%M:%S] [%s:%#] [%^%l%$] %v");

  std::string_view data_cfg =
      "/Users/mawenhui/Documents/darknet_cpp/cfg/coco.data";
  std::string_view cfgfile =
      "/Users/mawenhui/Documents/darknet_cpp/cfg/yolov3.cfg.toml";
  std::string_view weightfile =
      "/Users/mawenhui/Documents/darknet/yolov3.weights";
  std::string_view img_file =
      "/Users/mawenhui/Documents/darknet/data/dog.jpg";

  auto coco_names = get_coco_names(read_names(data_cfg));

  Network net(cfgfile, weightfile, true);
  net.SetBatch(1);

  cv::Mat origin_img =
      cv::imread(std::string(img_file), cv::ImreadModes::IMREAD_COLOR);
  cv::cvtColor(origin_img,origin_img,cv::COLOR_BGR2RGB);

  //prepare net input
  cv::Mat img(cv::Size(origin_img.cols,origin_img.rows),CV_32FC3);

  for (int k = 0; k < origin_img.channels(); k++) {
    for (int i = 0; i < origin_img.rows; i++) {
      for (int j = 0; j < origin_img.cols; j++) {
        img.at<cv::Vec3f>(i, j)[k] = origin_img.at<cv::Vec3b>(i, j)[k];
      }
    }
  }

  auto prepared = PrepareImage(img, net.w_, net.h_);
  Tensor<float> t(1, net.w_, net.h_, net.c_, NHWC, prepared);
  t.toNCHW();
  net.Forward(t);

  // box resize to raw image
  int new_w = 0;
  int new_h = 0;
  if (((float)net.w_ / origin_img.cols) < ((float)net.h_ / origin_img.rows)) {
    new_w = net.w_;
    new_h = (origin_img.rows * net.w_) / origin_img.cols;
  } else {
    new_h = net.h_;
    new_w = (origin_img.cols * net.h_) / origin_img.rows;
  }
  SPDLOG_INFO("new_w: {}, new_h: {}, net.w_: {}, net.h_: {}, w: {}, h: {}",
              new_w, new_h, net.w_, net.h_, origin_img.cols, origin_img.rows);
  for (auto& det : net.detections_) {
    box& b = det.bbox;
    b.x = (b.x - (net.w_ - new_w) / 2. / net.w_) / ((float)new_w / net.w_);
    b.y = (b.y - (net.h_ - new_h) / 2. / net.h_) / ((float)new_h / net.h_);
    b.w *= (float)net.w_ / new_w;
    b.h *= (float)net.h_ / new_h;
  }

  for (int i = 0; i < net.detections_.size() - 1; i++) {
    auto& det_a = net.detections_.at(i);
    if (det_a.objectness == 0) continue;
    for (int j = i + 1; j < net.detections_.size(); j++) {
      auto& det_b = net.detections_.at(i);
      if (det_b.objectness == 0) continue;
      auto iou = box_iou(det_a.bbox, det_b.bbox);
      if (iou > 0.5) {
        det_b.objectness = 0;
        std::fill(det_b.prob.begin(), det_b.prob.end(), 0.0);
      }
    }
  }

  std::sort(
      net.detections_.begin(), net.detections_.end(),
      [](detection a, detection b) { return a.objectness > b.objectness; });

  for (auto det : net.detections_) {
    auto max_prob = std::max_element(det.prob.begin(), det.prob.end());
    if (det.objectness != 0) {
      SPDLOG_INFO("[{},{},{},{}],objectness: {}, {}, prob: {}, sort_class: {}",
                  det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h,
                  det.objectness,
                  coco_names[std::distance(det.prob.begin(), max_prob)],
                  *max_prob, det.sort_class);
      auto b = det.bbox;

      int left = (b.x - b.w / 2.) * origin_img.cols;
      int right = (b.x + b.w / 2.) * origin_img.cols;
      int top = (b.y - b.h / 2.) * origin_img.rows;
      int bot = (b.y + b.h / 2.) * origin_img.rows;

      cv::rectangle(origin_img, cv::Rect(left, top, right - left, bot - top),
                    {0, 255, 0});
    }
  }
  cv::imwrite("result.jpg", origin_img);
  return 0;
}