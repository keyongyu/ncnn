//
// Created by keyong on 2019-03-25.
//

#include <stdio.h>
#include <vector>
#include <tuple>
#include <functional>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "net.h"
#include "cpu.h"

struct Object
{
    //cv::Rect_<float> rect;
    cv::Rect_<int> rect;
    int label;
    float prob;
};

struct Anchor{
    float center_x; //0.0 ~ 1.0
    float center_y; //[0, 1.0)
    float width;   //[0,1.0)
    float height;   //[0,1.0)
};
struct AnchorSpec{
    int feature_w;
    int feature_h;
    float stride;
    float scale;
    float aspect[3];

};
std::vector<Anchor>   generate_anchor()
{
//    const float image_w=600.0;
//    const float image_h=800.0;
//    AnchorSpec specs[5]={
//            {38, 50,  16, 0.035f,  {1.82940672f, 1.31881404f, 0.49710597f}},
//            {19, 25,  32, 0.08f,   {1.82940672f, 1.31881404f, 0.49710597f}},
//            {10, 13,  64, 0.16f,   {1.82940672f, 1.31881404f, 0.49710597f}},
//            {5,   7, 100, 0.32f,   {1.82940672f, 1.31881404f, 0.49710597f}},
//            {0,   0,  0,   0.6f,   {0.f, 0.f, 0.f}}
//    };
    const float image_w=400.0;
    const float image_h=533.0;
    AnchorSpec specs[5]={
            {25, 34, 16, 0.035f*3/2,  {1.82940672f, 1.31881404f, 0.49710597f}},
            {13, 17, 32, 0.08f*3/2,   {1.82940672f, 1.31881404f, 0.49710597f}},
            {7,  9,  64, 0.16f*3/2,   {1.82940672f, 1.31881404f, 0.49710597f}},
            {4,  5, 100, 0.32f*3/2,   {1.82940672f, 1.31881404f, 0.49710597f}},
            {0,   0,  0, 0.6f*3/2,    {0.f, 0.f, 0.f}}
    };
    std::vector<Anchor> anchor;

    const float image_aspect= image_w/image_h;
    for(int k=0; k < 5; k++) {
        AnchorSpec &spec = specs[k];
        if(spec.feature_h<=0)
            break;
        const float step_w = spec.stride / image_w;
        const float step_h = spec.stride / image_h;

        for (int j = 0; j < spec.feature_h; j++) {
            for (int i = 0; i < spec.feature_w; i++) {
                const float center_x = (i + 0.5f) * step_w;
                const float center_y = (j + 0.5f) * step_h;

                const float s_k= spec.scale;
                for (const float a : spec.aspect) {
                    if(a <=0.0)
                        break;
                    const float ar_sqrt= sqrt(a);
                    const float box_w = s_k * ar_sqrt;
                    const float box_h = s_k * image_aspect / ar_sqrt;
                    anchor.push_back({center_x, center_y, box_w, box_h});
                }
                auto s_k_prime = sqrt(s_k* specs[k+1].scale) ;
                anchor.push_back({center_x, center_y, s_k_prime, s_k_prime * image_aspect});
            } //end w
        }//end h
    }//end layer
    return anchor;
}

static cv::Rect_<float> decode_bbox(const float * loc, const Anchor& anch)
{
    const float center_variance=0.1;
    const float size_variance=0.2;
    cv::Rect_<float> rect;
    rect.x= loc[0]* center_variance* anch.width + anch.center_x;
    rect.y= loc[1]* center_variance* anch.height+ anch.center_y;
    rect.width= std::exp(loc[2]* size_variance)* anch.width;
    rect.height= std::exp(loc[3]* size_variance)* anch.height;
    rect.x= rect.x-rect.width/2;
    rect.y= rect.y-rect.height/2;
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

static inline int intersection_area(const cv::Rect_<int>& a, const cv::Rect_<int>& b)
{
    if (a.x > b.x+b.width || a.x+a.width < b.x || a.y > b.y+b.height || a.y+a.height < b.y)
    {
        return 0;
    }

    int inter_width = std::min(a.x+a.width, b.x+b.width) - std::max(a.x, b.x);
    int inter_height = std::min(a.y+a.height, b.y+b.height) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void nms(const std::vector<Object>& objects, std::vector<int>& picked, float iou_threshold)
{
    picked.clear();
    const size_t n = static_cast<const int>(objects.size());
    std::vector<float> areas( n );
    for (size_t i = 0; i < n; ++i)
    {
        const auto& r = objects[i].rect;
        areas[i] = r.width *r.height;
    }

    for (size_t i = 0; i < n; i++)
    {
        const auto& a = objects[i].rect;
        int keep = 1;
        for (int j : picked) {
            const auto& b = objects[j].rect;
            // intersection over union
            auto inter_area = intersection_area(a, b);
            const float iou = inter_area * 1.0f / std::min(areas[i],  areas[j]) ;
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

static int detect_mobilenetv2(const cv::Mat& bgr, std::vector<Object>& objects)
{
    int num_threads = ncnn::get_cpu_count()/2;
    if(num_threads==0)
        num_threads=1;

    //ncnn::set_cpu_powersave(2);
    //ncnn::Option opt;
    //opt.lightmode = true;
    //opt.num_threads = 4;
    //ncnn::set_default_option(opt);

    //0=all cores, 1=little cores only, 2=big cores only

    //ncnn::set_omp_dynamic(0);
    //ncnn::set_omp_num_threads(num_threads);

    auto anchors = generate_anchor();
    ncnn::Net mobilenetv2;

    //mobilenetv2.register_custom_layer("Silence", Noop_layer_creator);
#ifdef __ANDROID__
    mobilenetv2.load_param("/data/local/tmp/ncnn.param");
    mobilenetv2.load_model("/data/local/tmp/ncnn.bin");
#else
    mobilenetv2.load_param("/Users/keyong/Documents/gits/ncnn.git/cmake-build-debug/tools/onnx/ncnn.param");
    mobilenetv2.load_model("/Users/keyong/Documents/gits/ncnn.git/cmake-build-debug/tools/onnx/ncnn.bin");
#endif
    //mobilenetv2.load_param("pytorch_onnx_ncnn.param");
    //mobilenetv2.load_model("pytorch_onnx_ncnn.bin");
    const int target_width=  400;
    const int target_height= 533;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR,
            bgr.cols, bgr.rows, target_width, target_height);

    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    //const float norm_vals[3] = {1.0/128,1.0/128,1.0/128};
    in.substract_mean_normalize(mean_vals, nullptr);
    //ncnn::Mat in_chw;
    //transpose_nchw(in,in_chw);

    ncnn::Extractor ex = mobilenetv2.create_extractor();
    //ex.set_light_mode(true);
    ex.set_num_threads(4);

    int64 e1 = cv::getTickCount();
    //ex.input("data", in);
    ex.input(0, in);

    ncnn::Mat scores, loc;
    //class_num*3000 3000 is the number of anchors (19*19, 10x10, 5x5, 3x3, 2x2, 1x1)*6
    ex.extract("np_score",  scores);
    //ex.extract("439",  scores);
    int64 e2 = cv::getTickCount();
    //4*3000
    ex.extract("np_loc",  loc);
    int64 e3 = cv::getTickCount();
    double time1 = (e2 - e1)/ cv::getTickFrequency();
    double time2 = (e3 - e1)/ cv::getTickFrequency();
    fprintf(stderr, "detection takes %.4f + %0.4f = %0.4f seconds\n", time1, time2-time1,time2);

    int num_cls = scores.w;
    objects.clear();
    std::vector<Object> high_score_objs;

    for (int i=0; i<scores.h; i++)
    {
        const float* ptr= scores.row(i);
        int size = num_cls;
        for(int klass =1 ; klass < size; klass++){
            auto conf=ptr[klass];
            if(conf > 0.45 ){
                Object obj;
                obj.label=klass;
                obj.prob=conf;
                const Anchor& anch=anchors[i];
                const float * loc1=loc.row(i);
                auto rect = decode_bbox(loc1, anch);
                obj.rect.x= static_cast<int>(rect.x * img_w);
                obj.rect.y= static_cast<int>(rect.y * img_h);
                obj.rect.width= static_cast<int>(rect.width * img_w);
                obj.rect.height= static_cast<int>(rect.height * img_h);
                high_score_objs.push_back(obj);
            }
        }
    }
    std::vector<int> keep;
    if(!high_score_objs.empty()){
        //qsort_descent(high_score_objs, 0, static_cast<int>(high_score_objs.size() - 1));
        std::sort(high_score_objs.begin(), high_score_objs.end(), [] (const Object& a, const Object& b  )-> bool{
            return a.prob>b.prob;
        });
        nms(high_score_objs,keep,0.4);
        for(auto i_keep: keep)
        {
            objects.push_back(high_score_objs[i_keep]);
        }
    }
    int64 e4 = cv::getTickCount();
    double time3 = (e4 - e1)/ cv::getTickFrequency();
    fprintf(stderr, "detection takes %.4f + %0.4f + %0.4f = %0.4f seconds\n", time1, time2-time1 ,time3-time2, time3);
    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    cv::Mat image = bgr.clone();
    for(size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %d %d  %d x %.d\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0),3);

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
    cv::imwrite("result_pytorch_onnx_ncnn.jpg", image);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_mobilenetv2(m, objects);

    draw_objects(m, objects);

    return 0;
}
