#include "sam.h"

#include <onnxruntime_cxx_api.h>

#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <opencv2/opencv.hpp>
#include <vector>

struct SamModel {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions[2];
  std::unique_ptr<Ort::Session> sessionPre, sessionSam;
  std::vector<int64_t> inputShapePre, outputShapePre;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  bool bModelLoaded = false;
  std::vector<float> outputTensorValuesPre;

  const char *inputNamesSam[6]{"image_embeddings", "point_coords",   "point_labels",
                               "mask_input",       "has_mask_input", "orig_im_size"},
      *outputNamesSam[3]{"masks", "iou_predictions", "low_res_masks"};

  SamModel(const Sam::Parameter& param) {
    for (auto& p : param.models) {
      std::ifstream f(p);
      if (!f.good()) {
        std::cerr << "Model file " << p << " not found" << std::endl;
        return;
      }
    }

    for (int i = 0; i < 2; i++) {
      auto& provider = param.providers[i];
      auto& option = sessionOptions[i];

      option.SetIntraOpNumThreads(param.threadsNumber);
      option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

      if (provider.deviceType == 1) {
        OrtCUDAProviderOptions options;
        options.device_id = provider.gpuDeviceId;
        if (provider.gpuMemoryLimit > 0) {
          options.gpu_mem_limit = provider.gpuMemoryLimit;
        }
        option.AppendExecutionProvider_CUDA(options);
      }
    }

#if _MSC_VER
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    auto wpreModelPath = converter.from_bytes(param.models[0]);
    auto wsamModelPath = converter.from_bytes(param.models[1]);
#else
    auto wpreModelPath = param.models[0];
    auto wsamModelPath = param.models[1];
#endif

    sessionPre = std::make_unique<Ort::Session>(env, wpreModelPath.c_str(), sessionOptions[0]);
    if (sessionPre->GetInputCount() != 1 || sessionPre->GetOutputCount() != 1) {
      std::cerr << "Preprocessing model not loaded (invalid input/output count)" << std::endl;
      return;
    }

    sessionSam = std::make_unique<Ort::Session>(env, wsamModelPath.c_str(), sessionOptions[1]);
    if (sessionSam->GetInputCount() != 6 || sessionSam->GetOutputCount() != 3) {
      std::cerr << "Model not loaded (invalid input/output count)" << std::endl;
      return;
    }

    inputShapePre = sessionPre->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    outputShapePre = sessionPre->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (inputShapePre.size() != 4 || outputShapePre.size() != 4) {
      std::cerr << "Preprocessing model not loaded (invalid shape)" << std::endl;
      return;
    }

    bModelLoaded = true;
  }

  cv::Size getInputSize() const {
    if (!bModelLoaded) return cv::Size(0, 0);
    return cv::Size(inputShapePre[3], inputShapePre[2]);
  }
  bool loadImage(const cv::Mat& image) {
    std::vector<uint8_t> inputTensorValues(inputShapePre[0] * inputShapePre[1] * inputShapePre[2] *
                                           inputShapePre[3]);

    if (image.size() != cv::Size(inputShapePre[3], inputShapePre[2])) {
      std::cerr << "Image size not match" << std::endl;
      return false;
    }
    if (image.channels() != 3) {
      std::cerr << "Input is not a 3-channel image" << std::endl;
      return false;
    }

    for (int i = 0; i < inputShapePre[2]; i++) {
      for (int j = 0; j < inputShapePre[3]; j++) {
        inputTensorValues[i * inputShapePre[3] + j] = image.at<cv::Vec3b>(i, j)[2];
        inputTensorValues[inputShapePre[2] * inputShapePre[3] + i * inputShapePre[3] + j] =
            image.at<cv::Vec3b>(i, j)[1];
        inputTensorValues[2 * inputShapePre[2] * inputShapePre[3] + i * inputShapePre[3] + j] =
            image.at<cv::Vec3b>(i, j)[0];
      }
    }

    auto inputTensor = Ort::Value::CreateTensor<uint8_t>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShapePre.data(),
        inputShapePre.size());

    outputTensorValuesPre = std::vector<float>(outputShapePre[0] * outputShapePre[1] *
                                               outputShapePre[2] * outputShapePre[3]);
    auto outputTensorPre = Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValuesPre.data(), outputTensorValuesPre.size(),
        outputShapePre.data(), outputShapePre.size());

    const char *inputNamesPre[] = {"input"}, *outputNamesPre[] = {"output"};

    Ort::RunOptions run_options;
    sessionPre->Run(run_options, inputNamesPre, &inputTensor, 1, outputNamesPre, &outputTensorPre,
                    1);
    return true;
  }

  void getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints,
               const cv::Rect& roi, cv::Mat& outputMaskSam, double& iouValue) const {
    const size_t maskInputSize = 256 * 256;
    float maskInputValues[maskInputSize],
        hasMaskValues[] = {0},
        orig_im_size_values[] = {(float)inputShapePre[2], (float)inputShapePre[3]};
    memset(maskInputValues, 0, sizeof(maskInputValues));

    std::vector<float> inputPointValues, inputLabelValues;
    for (auto& point : points) {
      inputPointValues.push_back((float)point.x);
      inputPointValues.push_back((float)point.y);
      inputLabelValues.push_back(1);
    }
    for (auto& point : negativePoints) {
      inputPointValues.push_back((float)point.x);
      inputPointValues.push_back((float)point.y);
      inputLabelValues.push_back(0);
    }
    
    if (!roi.empty()) {
      inputPointValues.push_back((float)roi.x);
      inputPointValues.push_back((float)roi.y);
      inputLabelValues.push_back(2);
      inputPointValues.push_back((float)roi.br().x);
      inputPointValues.push_back((float)roi.br().y);
      inputLabelValues.push_back(3);
    }

    const int numPoints = inputLabelValues.size();
    std::vector<int64_t> inputPointShape = {1, numPoints, 2}, pointLabelsShape = {1, numPoints},
                         maskInputShape = {1, 1, 256, 256}, hasMaskInputShape = {1},
                         origImSizeShape = {2};

    std::vector<Ort::Value> inputTensorsSam;
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)outputTensorValuesPre.data(), outputTensorValuesPre.size(),
        outputShapePre.data(), outputShapePre.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputPointValues.data(),
                                                              2 * numPoints, inputPointShape.data(),
                                                              inputPointShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputLabelValues.data(),
                                                              numPoints, pointLabelsShape.data(),
                                                              pointLabelsShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));

    Ort::RunOptions runOptionsSam;
    auto outputTensorsSam = sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(),
                                            inputTensorsSam.size(), outputNamesSam, 3);

    auto outputMasksValues = outputTensorsSam[0].GetTensorMutableData<float>();
    if (outputMaskSam.type() != CV_8UC1 ||
        outputMaskSam.size() != cv::Size(inputShapePre[3], inputShapePre[2])) {
      outputMaskSam = cv::Mat(inputShapePre[2], inputShapePre[3], CV_8UC1);
    }

    for (int i = 0; i < outputMaskSam.rows; i++) {
      for (int j = 0; j < outputMaskSam.cols; j++) {
        outputMaskSam.at<uchar>(i, j) = outputMasksValues[i * outputMaskSam.cols + j] > 0 ? 255 : 0;
      }
    }

    iouValue = outputTensorsSam[1].GetTensorMutableData<float>()[0];
  }
};

Sam::Sam(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber)
    : Sam(Parameter(preModelPath, samModelPath, threadsNumber)) {}

Sam::Sam(const Parameter& param) : m_model(new SamModel(param)) {}

Sam::~Sam() { delete m_model; }

cv::Size Sam::getInputSize() const { return m_model->getInputSize(); }
bool Sam::loadImage(const cv::Mat& image) { return m_model->loadImage(image); }

cv::Mat Sam::getMask(const cv::Point& point, double* iou) const {
  return getMask({point}, {}, {}, iou);
}

cv::Mat Sam::getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints,
                     double* iou) const {
  return getMask(points, negativePoints, {}, iou);
}

cv::Mat Sam::getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints,
                     const cv::Rect& roi, double* iou) const {
  double iouValue = 0;
  cv::Mat m;
  m_model->getMask(points, negativePoints, roi, m, iouValue);
  if (iou != nullptr) {
    *iou = iouValue;
  }
  return m;
}

// Just a poor version of
// https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
cv::Mat Sam::autoSegment(const cv::Size& numPoints, cbProgress cb, const double iouThreshold,
                         const double minArea, int* numObjects) const {
  if (numPoints.empty()) {
    return {};
  }

  const auto size = getInputSize();
  cv::Mat mask, outImage = cv::Mat::zeros(size, CV_64FC1);

  std::vector<double> masksAreas;

  for (int i = 0; i < numPoints.height; i++) {
    for (int j = 0; j < numPoints.width; j++) {
      if (cb) {
        cb(double(i * numPoints.width + j) / (numPoints.width * numPoints.height));
      }

      cv::Point input(cv::Point((j + 0.5) * size.width / numPoints.width,
                                (i + 0.5) * size.height / numPoints.height));

      double iou;
      m_model->getMask({input}, {}, {}, mask, iou);
      if (mask.empty() || iou < iouThreshold) {
        continue;
      }

      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // find segmented region for 1 mask at a time
      if (contours.empty()) {
        continue;
      }

      int maxContourIndex = 0;
      double maxContourArea = 0;
      for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxContourArea) {
          maxContourArea = area;
          maxContourIndex = i;
        }
      }
      if (maxContourArea < minArea) {
        continue;
      }

      cv::Mat contourMask = cv::Mat::zeros(size, CV_8UC1);
      cv::drawContours(contourMask, contours, maxContourIndex, cv::Scalar(255), cv::FILLED);
      cv::Rect boundingBox = cv::boundingRect(contours[maxContourIndex]);

      int index = masksAreas.size() + 1, numPixels = 0;
      for (int i = boundingBox.y; i < boundingBox.y + boundingBox.height; i++) {
        for (int j = boundingBox.x; j < boundingBox.x + boundingBox.width; j++) {
          if (contourMask.at<uchar>(i, j) == 0) {
            continue;
          }

          auto dst = (int)outImage.at<double>(i, j);
          if (dst > 0 && masksAreas[dst - 1] < maxContourArea) {
            continue;
          }
          outImage.at<double>(i, j) = index;
          numPixels++;
        }
      }
      if (numPixels == 0) {
        continue;
      }

      masksAreas.emplace_back(maxContourArea);
    }
  }
  return outImage;
}


// Just a poor version of
// https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
// return contours vector<vector<Point>>
std::vector<std::vector<cv::Point>> Sam::autoSegmentContour(const cv::Size& numPoints, cbProgress cb, const double iouThreshold,
    const double minArea, int* numObjects) const {
    if (numPoints.empty()) {
        return {};
    }

    const auto size = getInputSize();
    cv::Mat mask, outImage = cv::Mat::zeros(size, CV_64FC1);

    std::vector<double> masksAreas;
    std::vector<std::vector<cv::Point>> objectContours; // points at the edge of segment masks
    for (int i = 0; i < numPoints.height; i++) {
        for (int j = 0; j < numPoints.width; j++) {
            if (cb) {
                cb(double(i * numPoints.width + j) / (numPoints.width * numPoints.height));
            }

            cv::Point input(cv::Point((j + 0.5) * size.width / numPoints.width,
                (i + 0.5) * size.height / numPoints.height));

            double iou;
            m_model->getMask({ input }, {}, {}, mask, iou);
            if (mask.empty() || iou < iouThreshold) {
                continue;
            }

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (contours.empty()) {
                continue;
            }

            int maxContourIndex = 0;
            double maxContourArea = 0;
            for (int i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > maxContourArea) {
                    maxContourArea = area;
                    maxContourIndex = i;
                }
            }
            if (maxContourArea < minArea) {
                continue;
            }
            
            //cv::namedWindow("Contours", cv::WINDOW_AUTOSIZE);
            //cv::Mat contoursImage = cv::Mat::zeros(mask.size(), CV_8UC3);
            //cv::Scalar color(0, 0, 255); // Red color for contours
            //cv::drawContours(contoursImage, contours, -1, color, 2); // -1表示绘制所有的轮廓
            //cv::imshow("Contours", contoursImage);
            //cv::waitKey(0); // 等待用户按键

            objectContours.push_back(contours[maxContourIndex]);

            //cv::namedWindow("Contours", cv::WINDOW_AUTOSIZE);
            //cv::Mat contoursImage = cv::Mat::zeros(mask.size(), CV_8UC3);
            //cv::Scalar color(0, 0, 255); // Red color for contours
            //cv::drawContours(contoursImage, objectContours, -1, color, 2); // -1表示绘制所有的轮廓
            //cv::imshow("Contours", contoursImage);
            //cv::waitKey(0); // 等待用户按键

        }
    }
    // points at the edge of images should be eliminated
    std::vector<std::vector<cv::Point>> centerContours; 
    int distanceThreshold = 15;

    for (auto& contour : objectContours) {
        std::vector<cv::Point> filteredContour; // 用于存储过滤后的轮廓点

        for (const cv::Point& point : contour) {
            int x = point.x;
            int y = point.y;

            // 检查点是否足够远离图像边缘
            if (x >= distanceThreshold && y >= distanceThreshold &&
                x < size.width - distanceThreshold && y < size.height - distanceThreshold) {
                filteredContour.push_back(point); // 如果点足够远离边缘，将其加入过滤后的轮廓
            }
        }

        if (!filteredContour.empty()) {
            contour = filteredContour; // 更新轮廓为过滤后的轮廓
        }
        else {
            // 如果过滤后的轮廓为空，可以选择将轮廓从列表中删除
            objectContours.erase(std::remove(objectContours.begin(), objectContours.end(), contour), objectContours.end());
        }
    }

    return objectContours;
}