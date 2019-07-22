//
// Created by keyong on 2019-03-25.
//
#include <stdio.h>
#include <vector>
#include <tuple>
#include <functional>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "net.h"
#include "cpu.h"

using namespace std;

struct Object {
    //cv::Rect_<float> rect;
    cv::Rect_<int> rect;
    int label;
    float prob;
};

struct Anchor {
    float center_x; //0.0 ~ 1.0
    float center_y; //[0, 1.0)
    float width;   //[0,1.0)
    float height;   //[0,1.0)
};

struct AnchorSpec {
    int feature_w;
    int feature_h;
    float step_w;
    float step_h;
    float scale;
    vector<float> aspect;
};

struct TrainParam {
    string version;
    vector<float> anchor_width_per_layer;
    vector<float> aspects_per_anchor;
    int num_class;

    TrainParam() : num_class(0) {}
};

TrainParam load_train_param(FILE *fp) {
    TrainParam tp;
    char version[100];
    float temp;
    if (1 != fscanf(fp, "Version: %99s\n", version) || 0 != strcmp(version, "MBV2_1")) {
        return tp;
    }
    tp.version = version;
    if (1 != fscanf(fp, "StepScale: %f", &temp)) {
        return tp;
    }
    tp.anchor_width_per_layer.push_back(temp);
    while (1 == fscanf(fp, "%f", &temp))
        tp.anchor_width_per_layer.push_back(temp);

    if (1 != fscanf(fp, "AspectRatio: %f\n", &temp))
        return tp;
    tp.aspects_per_anchor.push_back(temp);
    while (1 == fscanf(fp, "%f", &temp))
        tp.aspects_per_anchor.push_back(temp);

    if (1 != fscanf(fp, "SkuNum: %d\n", &tp.num_class))
        return tp;
    //skip content
    fgets(version, sizeof(version), fp);
    return tp;
}

vector<AnchorSpec> make_anchor_spec(const int img_width, const int img_height,
                                    const TrainParam &train_param) {
    int feature_h = (img_height + 15) / 16;
    int feature_w = (img_width + 15) / 16;
    float step_h = 16.0f / img_height;
    float step_w = 16.0f / img_width;
    const int num_layers = train_param.anchor_width_per_layer.size();
    vector<AnchorSpec> specs(num_layers);

    //skip last layer, the anchor width of last layer is used for compensation only
    for (int l = 0; l < num_layers - 1; l++) {
        AnchorSpec &spec = specs[l];
        spec.aspect = train_param.aspects_per_anchor;

        spec.feature_h = feature_h;
        spec.feature_w = feature_w;
        spec.scale = train_param.anchor_width_per_layer[l] * 600 / img_width;
        spec.step_w = step_w;
        spec.step_h = step_h;
        step_h *= 2.0;
        step_w *= 2.0;

        feature_h = (feature_h + 1) / 2;
        feature_w = (feature_w + 1) / 2;
    }
    if (num_layers > 1)
        specs[num_layers - 1].scale = train_param.anchor_width_per_layer[num_layers - 1] * 600 / img_width;

    return specs;
}


std::vector<Anchor> generate_anchor(const vector<AnchorSpec> &specs,
                                    const int image_w, const int image_h) {
    std::vector<Anchor> anchor;
    const float image_aspect = image_w * 1.0 / image_h;
    for (size_t k = 0; k < specs.size(); k++) {
        auto &spec = specs[k];
        if (spec.feature_h <= 0)
            break;

        const float step_w = spec.step_w;
        const float step_h = spec.step_h;


        for (int j = 0; j < spec.feature_h; j++) {
            for (int i = 0; i < spec.feature_w; i++) {

                float center_x = (i + 0.5f) * step_w;
                const float center_y = (j + 0.5f) * step_h;
                const float s_k = spec.scale;
                for (const float a : spec.aspect) {
                    const float ar_sqrt = sqrt(a);
                    const float box_w = s_k * ar_sqrt;
                    const float box_h = s_k * image_aspect / ar_sqrt;
                    if (a > 0.333) {
                        anchor.push_back({center_x, center_y, box_w, box_h});
                    } else {
                        float x1 = center_x - 0.5 * s_k;
                        float x2 = center_x + 0.5 * s_k;
                        while (x1 + box_w <= x2) {
                            anchor.push_back({x1 + box_w / 2, center_y, box_w, box_h});
                            x1 = x1 + box_w;
                        }
                        if (x1 < x2 && x1 + box_w > x2) {
                            anchor.push_back({x2 - box_w / 2, center_y, box_w, box_h});
                        }
                    }
                }
                auto s_k_prime = sqrt(s_k * specs[k + 1].scale);
                anchor.push_back({center_x, center_y, s_k_prime, s_k_prime * image_aspect});
            } //end w
        }//end h
    }//end layer
    return anchor;
}

static cv::Rect_<float> decode_bbox(const float *loc, const Anchor &anch) {
    const float center_variance = 0.1;
    const float size_variance = 0.2;
    cv::Rect_<float> rect;
    rect.x = loc[0] * center_variance * anch.width + anch.center_x;
    rect.y = loc[1] * center_variance * anch.height + anch.center_y;
    rect.width = std::exp(loc[2] * size_variance) * anch.width;
    rect.height = std::exp(loc[3] * size_variance) * anch.height;
    rect.x = rect.x - rect.width / 2;
    rect.y = rect.y - rect.height / 2;
    return rect;
}

//static void qsort_descent(std::vector<Object>& objects, int left, int right)
//{
//    int i = left;
//    int j = right;
//    if(i+1 == j){
//        if (objects[i].prob < objects[j].prob)
//            std::swap(objects[i], objects[j]);
//        return;
//    }
//    const float p = objects[(left + right) / 2].prob;
//    while (i < j)
//    {
//        while (objects[i].prob > p)
//            i++;
//
//        while (objects[j].prob < p)
//            j--;
//
//        if (i < j)
//        {
//            // swap
//            std::swap(objects[i], objects[j]);
//            i++;
//            j--;
//        }
//    }
//
//    if (left < j)
//        qsort_descent(objects, left, j);
//
//    if (i < right)
//        qsort_descent(objects, i, right);
//}

static inline int intersection_area(const cv::Rect_<int> &a, const cv::Rect_<int> &b) {
    if (a.x > b.x + b.width || a.x + a.width < b.x || a.y > b.y + b.height || a.y + a.height < b.y) {
        return 0;
    }

    int inter_width = std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x);
    int inter_height = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void nms(const std::vector<Object> &objects, std::vector<int> &picked, float iou_threshold) {
    picked.clear();
    const size_t n = static_cast<const int>(objects.size());
    std::vector<float> areas(n);
    for (size_t i = 0; i < n; ++i) {
        const auto &r = objects[i].rect;
        areas[i] = r.width * r.height;
    }

    for (size_t i = 0; i < n; i++) {
        const auto &a = objects[i].rect;
        int keep = 1;
        for (int j : picked) {
            const auto &b = objects[j].rect;
            // intersection over union
            auto inter_area = intersection_area(a, b);
            const float iou = inter_area * 1.0f / std::min(areas[i], areas[j]);
            if (iou > iou_threshold)
                keep = 0;
            //auto union_area = areas[i] + areas[j] - inter_area;
            //if (union_area==0  || inter_area> union_area* iou_threshold )
            //    keep = 0;

        }

        if (keep)
            picked.push_back(static_cast<int>(i));
    }
}

static int detect_mobilenetv2(const cv::Mat &bgr, std::vector<Object> &objects,
                              ncnn::Extractor &ex, const TrainParam &train_par) {
    //int64 e1 = cv::getTickCount();

    int target_width, target_height;
    if (bgr.rows > bgr.cols) {
        target_width = 600;
        target_height = bgr.rows * 600 / bgr.cols;
    } else {
        //target_width = 800;
        //target_height = bgr.rows * 800 / bgr.cols;
        target_height = 600;
        target_width = bgr.cols * 600 / bgr.rows;
    }
    auto anchor_specs = make_anchor_spec(target_width, target_height, train_par);
    auto anchors = generate_anchor(anchor_specs, target_width, target_height);
    //int num_threads = ncnn::get_cpu_count()/2;
    //if(num_threads==0)
    //    num_threads=1;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR,
                                                 bgr.cols, bgr.rows, target_width, target_height);

    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    //const float norm_vals[3] = {1.0/128,1.0/128,1.0/128};
    in.substract_mean_normalize(mean_vals, nullptr);

    //ncnn::Extractor ex = mobilenetv2.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    //int64 e1 = cv::getTickCount();
    //ex.input("data", in);
    ex.input(0, in);

    ncnn::Mat scores, loc;
    //class_num*3000 3000 is the number of anchors (19*19, 10x10, 5x5, 3x3, 2x2, 1x1)*6
    ex.extract("np_score", scores);
    //ex.extract("439",  scores);
    //int64 e2 = cv::getTickCount();
    //4*3000
    ex.extract("np_loc", loc);
    //int64 e3 = cv::getTickCount();
    //double time1 = (e2 - e1) / cv::getTickFrequency();
    //double time2 = (e3 - e1) / cv::getTickFrequency();
    //fprintf(stderr, "detection takes %.4f + %0.4f = %0.4f seconds\n", time1, time2 - time1, time2);

    int num_cls = scores.w;
    objects.clear();
    std::vector<Object> high_score_objs;

    for (int i = 0; i < scores.h; i++) {
        const float *ptr = scores.row(i);
        int size = num_cls;
        for (int klass = 1; klass < size; klass++) {
            auto conf = ptr[klass];
            if (conf > 0.45 /*&& klass==49*/) {
                Object obj;
                obj.label = klass;
                obj.prob = conf;
                const Anchor &anch = anchors[i];
                const float *loc1 = loc.row(i);
                auto rect = decode_bbox(loc1, anch);
                obj.rect.x = static_cast<int>(rect.x * img_w);
                obj.rect.y = static_cast<int>(rect.y * img_h);
                obj.rect.width = static_cast<int>(rect.width * img_w);
                obj.rect.height = static_cast<int>(rect.height * img_h);
                high_score_objs.push_back(obj);
            }
        }
    }
    std::vector<int> keep;
    if (!high_score_objs.empty()) {
        //qsort_descent(high_score_objs, 0, static_cast<int>(high_score_objs.size() - 1));
        std::sort(high_score_objs.begin(), high_score_objs.end(), [](const Object &a, const Object &b) -> bool {
            return a.prob > b.prob;
        });
        nms(high_score_objs, keep, 0.3);
        //nms(high_score_objs, keep, 1.0);
        for (auto i_keep: keep) {
            objects.push_back(high_score_objs[i_keep]);
        }
    }
    //int64 e4 = cv::getTickCount();
    //double time3 = (e4 - e1) / cv::getTickFrequency();
    //fprintf(stderr, "detection takes %.4f + %0.4f + %0.4f = %0.4f seconds\n", time1, time2 - time1, time3 - time2,
    //        time3);
    return 0;
}

static void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects, const string &new_image_name) {
    cv::Mat image = bgr.clone();
    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        /*fprintf(stderr, "%d = %.5f at %d %d  %d x %.d\n", obj.label, obj.prob,
                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);*/

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0), 3);

        char text[256];
        //sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        sprintf(text, "%d %.1f%%", obj.label, obj.prob * 100);

        int baseLine = 10;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine + 20;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), CV_FILLED);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 2);
    }

    //cv::imshow("image", image);
    //cv::waitKey(0);
    //cv::imwrite("result_pytorch_onnx_ncnn.jpg", image);
    cv::imwrite(new_image_name.c_str(), image);
}

int main(int /*argc*/, char **argv) {
    //if (argc != 2)
    //{
    //    fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
    //    return -1;
    //}

    // const char* imagepath = argv[1];



    ncnn::Net mobilenetv2;
    FILE *fp = fopen(argv[1], "rb");
    auto train_par = load_train_param(fp);
    assert(train_par.version == "MBV2_1");
    assert(train_par.anchor_width_per_layer.size() >= 3);
    assert(train_par.aspects_per_anchor.size() >= 1);
    mobilenetv2.load_param(fp);
    //auto layer = mobilenetv2.find_layer_by_name("664");
    //name 664, type: "Reshape", w:52=(51+1)
    //todo, check the number of sku type;
    //auto num_kind= mobilenetv2.



    std::string imagepath = argv[2];
    //std::string short_name= strrchr(argv[2],'/');
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", imagepath.c_str());
        return 0;
    }
    mobilenetv2.load_model(fp);
    fclose(fp);

    ncnn::Extractor ex = mobilenetv2.create_extractor();
    std::vector<Object> objects;
    detect_mobilenetv2(m, objects, ex, train_par);
    draw_objects(m, objects, imagepath + "_result.jpg");
    return 0;
}


