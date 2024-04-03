//
// Created by zhu on 2023/2/21.
//

#include <include/PythonClient.h>
#include <iostream>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/highgui.hpp>
using namespace std;

namespace ORB_SLAM3 {

PythonClient::PythonClient(int port) {

    //创建一个socket
    tcpserver = socket(AF_INET, SOCK_STREAM, 0);

    //    准备通讯地址
    struct sockaddr_in serveraddr;
    serveraddr.sin_family = AF_INET;
    serveraddr.sin_port = htons(port);
    serveraddr.sin_addr.s_addr = inet_addr("127.0.0.1");

    cout << "bind ok 等待客户端的连接" << endl;

    while (true) {
        if (connect(tcpserver, (struct sockaddr *) &serveraddr, sizeof(serveraddr)) == 0);
        break;
    }

}

void PythonClient::RecvData(cv::Mat &img) {
    // 单通道要乘2，不知道为什么
    int nBytes = mImage.rows * mImage.cols * 2;

    auto pBuf = new uchar[nBytes];
    uchar *pData = pBuf;

    while (nBytes) {

        int recvBytes;
        if (nBytes > 65536)
            recvBytes = recv(tcpserver, pData, 65536, 0);
        else
            recvBytes = recv(tcpserver, pData, nBytes, 0);
        pData += recvBytes;
        nBytes -= recvBytes;
    }
    img = cv::Mat(mImage.rows, mImage.cols, CV_16UC1, pBuf).clone();
//    cout<<img.type()<<endl;
//    cout<< sizeof(img)<<endl;

    delete[] pBuf;
    pBuf = nullptr;
}


void PythonClient::SendData() {
    int nBytes;
    assert(mImage.type() == CV_8UC1 || mImage.type() == CV_8UC3);
    if (mImage.type() == CV_8UC3)
        nBytes = mImage.rows * mImage.cols * 3;
    else
        nBytes = mImage.rows * mImage.cols;

    int nTimes = nBytes / 65536, nRes = nBytes - nTimes * 65536;
    uchar *pData = mImage.data;

    for (int i = 0; i < nTimes; i++) {
        send(tcpserver, pData, 65536, 0);
        pData = pData + 65536;
    }
    if (nRes > 0)
        send(tcpserver, pData, nRes, 0);
}

void PythonClient::GetDepthImage(const cv::Mat &image, cv::Mat &depthmap) {
    image.copyTo(mImage);
    SendData();
    RecvData(depthmap);

//    cv::Mat obj_image;
//    RecvObj_Detect(obj_image);
//    cv::imshow("obj", obj_image);
//    cv::waitKey(1);
}

//void PythonClient::ObjectDetect(const cv::Mat &image, cv::Mat &obj_image) {
//    image.copyTo(mImage);
//    SendData();
//    RecvObj_Detect(obj_image);
//}

void PythonClient::show()
{
    Imgtest = cv::imread("/home/zhu/桌面/111.jpeg", 1);
    cv::imshow("test", Imgtest);
    cv::waitKey(0);
}

PythonClient::~PythonClient() {
    close(tcpserver);

}
}