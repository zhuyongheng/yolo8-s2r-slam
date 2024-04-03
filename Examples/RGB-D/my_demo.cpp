//
// Created by zhu on 2023/2/23.
//

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <PythonClient.h>
#include <imageLoader.h>

#include<System.h>

using namespace cv;
using namespace std;

//void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft,
//                vector<string> &vstrImageRight, vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
//    参数设置
    const string BoWFilePath = "../../Vocabulary/ORBvoc.txt";
    const string settingFilePath = "./P05.yaml";

//    加载数据集
    DatasetsLoader imageLoader(argv[1]);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(BoWFilePath,settingFilePath,ORB_SLAM3::System::STEREO,true);

    // 主循环
    cv::Mat imLeft, imRight, imD;
    string fileName;
    int timestep = 0;
    while(true)
    {
        if(imageLoader.LoadStereoImage())
        imLeft = imageLoader.LeftImage();
        imRight = imageLoader.RightImage();
        imageLoader.GetFileName(fileName);

        ORB_SLAM3::KeyFrame::SetFileName(fileName);

        SLAM.TrackStereo(imLeft, imRight, timestep);
        timestep = timestep + 1;
    }



}


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{





}






