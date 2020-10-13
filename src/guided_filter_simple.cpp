
#include "guided_filter_simple.h"

void compute(std::string input_file, std::string g_file, std::string output_file) {
    cv::Mat input = cv::imread(input_file);
    if(input.empty()) {
        std::cout<<"Input image Not Found: "<< input_file << std::endl;
        return;
    }
    cv::Mat g_original = cv::imread(g_file);
    if(g_original.empty()) {
        std::cout<<"guidance image Not Found: "<< g_file << std::endl;
        return;
    }

    cv::Mat g;
    cvtColor(g_original, g, CV_BGR2RGBA, 4);
    g.convertTo(g, CV_32FC4);
    g /= 255.f;

    // cv::Mat p = inputRGBA.clone();
    cv::Mat p;
    cvtColor(input, p, CV_BGR2RGBA, 4);
    p.convertTo(p, CV_32FC4);
    p /= 255.f;

    cv::Mat output (input.size(), g.type());

    float eps = 0.2 * 0.2;
    cv::Mat tmp = g.mul(p);
    cv::Mat tmp2 = g.mul(g);

    guided_filter_cuda(g.ptr<float4>(),
            p.ptr<float4>(),
            output.ptr<float4>(),
            tmp.ptr<float4>(),
            tmp2.ptr<float4>(),
            g.cols, g.rows,
            eps);

    output *= 255;

    cvtColor(output, output, CV_RGBA2BGR, 3);

    imwrite(output_file, output);
    printf("Saved image: %s\n", output_file.c_str());
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Choose image" << std::endl;
        return 1;
    }
    compute(argv[argc - 2], argv[argc - 1], "out.png");
    return 0;
}
