//
// Created by Administrator on 2022/3/9.
//

#include "picodet.h"

using namespace std;

PicoDet::PicoDet(const std::string& mnn_path, int input_width, int input_length, int num_thread_,
    float score_threshold_, float nms_threshold_) {
    num_thread = num_thread_;
    in_w = input_width;
    in_h = input_length;
    score_threshold = score_threshold_;
    nms_threshold = nms_threshold_;

    interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.numThread = num_thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)2;
    scheduleConfig.backendConfig = &backendConfig;

    session = interpreter->createSession(scheduleConfig);
    input_tensor = interpreter->getSessionInput(session, nullptr);
    input_tensor->printShape();
}

PicoDet::~PicoDet() {
    interpreter->releaseModel();
    interpreter->resizeSession(session);
}

int PicoDet::detect(cv::Mat& raw_img, std::vector<BoxInfo>& result_list) {
    if (raw_img.empty()) {
        std::cout << "image is empty, please check!" << std::endl;
        return -1;
    }

    image_h = raw_img.rows;
    image_w = raw_img.cols;

    cv::Mat image;

    cv::resize(raw_img, image, cv::Size(in_w, in_h));

    interpreter->resizeTensor(input_tensor, { 1, 3, in_h, in_w });
    interpreter->resizeSession(session);

    MNN::CV::Matrix trans;
    trans.setScale(1.0f, 1.0f);
    MNN::CV::ImageProcess::Config config;
    config.filterType = MNN::CV::BICUBIC;
    config.sourceFormat = MNN::CV::BGR;
    config.destFormat = MNN::CV::BGR;
    ::memcpy(config.mean, mean_vals, sizeof(mean_vals));
    ::memcpy(config.normal, norm_vals, sizeof(norm_vals));

    pretreat = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(config));
    pretreat->setMatrix(trans);
    //pretreat = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::BGR, mean_vals, 3, norm_vals, 3));
    pretreat->convert(image.data, in_w, in_h, image.step[0], input_tensor);

    auto start = chrono::steady_clock::now();

    // run network
    interpreter->runSession(session);

    // get output data
    std::vector<std::vector<BoxInfo>> results;

    results.resize(num_class);


    for (const auto& head_info : heads_info) {
        MNN::Tensor* tensor_scores = interpreter->getSessionOutput(
            session, head_info.cls_layer.c_str());

        MNN::Tensor* tensor_boxes = interpreter->getSessionOutput(
            session, head_info.dis_layer.c_str()
        );

        MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
        tensor_scores->copyToHostTensor(&tensor_scores_host);
        MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
        tensor_boxes->copyToHostTensor(&tensor_boxes_host);

        decode_infer(&tensor_scores_host, &tensor_boxes_host, head_info.stride, score_threshold, results);
    }

    auto end = chrono::steady_clock::now();

    chrono::duration<double> elapsed = end - start;

    cout << "inference time:" << elapsed.count() << " s, ";

    for (int i = 0; i < (int)results.size(); ++i) {
        nms(results[i], nms_threshold);
        for (auto box : results[i]) {
            box.x1 = box.x1 / in_w * image_w;
            box.x2 = box.x2 / in_w * image_w;
            box.y1 = box.y1 / in_h * image_h;
            box.y2 = box.y2 / in_h * image_h;
            result_list.push_back(box);
        }
    }
    cout << "detect " << result_list.size() << " objects" << endl;

    return 0;
}


void PicoDet::decode_infer(MNN::Tensor* cls_pred, MNN::Tensor* dis_pred, int stride, float threshold,
    vector<std::vector<BoxInfo>>& results) {

    int feature_h = ceil((float)in_h / stride);
    int feature_w = ceil((float)in_w / stride);

    for (int idx = 0; idx < feature_h * feature_w; ++idx) {
        const float* scores = cls_pred->host<float>() + (idx * num_class);
        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; ++label) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            const float* bbox_pred = dis_pred->host<float>() + (idx * 4 * (reg_max + 1));
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
        }
    }
}

BoxInfo PicoDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride) {
    float ct_x = (x + 0.5) * stride;
    float ct_y = (y + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm,
            reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)in_w);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)in_h);
    return BoxInfo{ xmin, ymin, xmax, ymax, score, label };
}

void PicoDet::nms(vector<BoxInfo>& input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(),
        [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
            (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}


std::string PicoDet::get_label_str(int label) {
    return labels[label];
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};

    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }


template <typename Tp>
int activation_function_softmax(const Tp* src, Tp* dst, int length) {
    const Tp alpha = *std::max_element(src, src + length);
    Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    return 0;
}