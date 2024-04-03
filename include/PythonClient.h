//
// Created by zhu on 2023/2/21.
//
#ifndef DENSE_MAPPING_PYTHONCLIENT_H
#define DENSE_MAPPING_PYTHONCLIENT_H


#include <opencv2/core.hpp>

#include <sys/socket.h>
#include <thread>
#include <mutex>
#include <list>

namespace  ORB_SLAM3 {

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class GeometricCamera;
class KeyFrame;

class PythonClient {
public:
    explicit PythonClient(int port);

    ~PythonClient();

    void GetDepthImage(const cv::Mat &image, cv::Mat &depthmap);
    void ObjectDetect(const cv::Mat &image, cv::Mat &obj_image);


    void show();

    float mDepthMapFactor;



private:
    int tcpserver = -1;
    cv::Mat mImage;
    cv::Mat Imgtest;

    void SendData();

    void RecvData(cv::Mat &img);

    void RecvObj_Detect(cv::Mat &img);

//    PythonClient* mpPythonClient;

};

}
#endif //DENSE_MAPPING_PYTHONCLIENT_H