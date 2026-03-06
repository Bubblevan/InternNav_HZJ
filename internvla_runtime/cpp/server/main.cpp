#include <iostream>
#include <grpcpp/grpcpp.h>
#include "internvla.grpc.pb.h"

#include <opencv2/opencv.hpp>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using internvla::InternVLAService;
using internvla::StepRequest;
using internvla::StepResponse;

class InternVLAServiceImpl final : public InternVLAService::Service {

public:

  Status Step(ServerContext* context,
              const StepRequest* request,
              StepResponse* response) override {

    int w = request->width();
    int h = request->height();

    const std::string& rgb_bytes = request->rgb();
    const std::string& depth_bytes = request->depth();

    cv::Mat rgb(h, w, CV_8UC3, (void*)rgb_bytes.data());
    cv::Mat depth(h, w, CV_32F, (void*)depth_bytes.data());

    cv::Mat rgb_resized;
    cv::resize(rgb, rgb_resized, cv::Size(384,384));

    cv::Mat depth_resized;
    cv::resize(depth, depth_resized, cv::Size(384,384));

    // normalize
    rgb_resized.convertTo(rgb_resized, CV_32F, 1.0/255.0);

    // flatten tensor
    std::vector<float> rgb_tensor;
    rgb_tensor.assign((float*)rgb_resized.data,
                      (float*)rgb_resized.data + 384*384*3);

    std::vector<float> depth_tensor;
    depth_tensor.assign((float*)depth_resized.data,
                        (float*)depth_resized.data + 384*384);

    // TODO: send to python worker
    // mock result

    response->set_has_action(true);
    response->add_action(1);

    return Status::OK;
  }
};

void RunServer() {

  std::string address("0.0.0.0:50051");

  InternVLAServiceImpl service;

  ServerBuilder builder;

  builder.AddListeningPort(address,
      grpc::InsecureServerCredentials());

  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());

  std::cout << "Server listening on " << address << std::endl;

  server->Wait();
}

int main() {

  RunServer();

  return 0;
}