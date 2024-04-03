//
// Created by zhu on 2023/2/23.
//

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>
#include <PythonClient.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft, vector<string> &vstrImageRight);
void LoadImages_new(const string &strAssociationFilename, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    std::chrono::steady_clock::time_point beforetime = std::chrono::steady_clock::now();

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;

    string strAssociationFilename = string(argv[4]);
    LoadImages_new(strAssociationFilename, vstrImageLeft, vstrImageRight, vTimestamps);

    int nImages  = vstrImageLeft.size();


    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::STEREO,true);

    // 主循环
    cv::Mat imLeft, imRight;

    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(string(argv[3])+"/"+vstrImageLeft[ni],cv::IMREAD_UNCHANGED);
        imRight = cv::imread(string(argv[3])+"/"+vstrImageRight[ni],cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];


        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }


        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft, imRight, tframe);

        // Wait to load the next frame
//            usleep(100000);

    }

    // Stop all threads
//    SLAM.save();
    SLAM.Shutdown();

//    保存相机轨迹
//    SLAM.SaveTrajectoryKITTI("seq_Trajectory.txt");

    std::chrono::steady_clock::time_point afterTime = std::chrono::steady_clock::now();
    double alltime= std::chrono::duration_cast<std::chrono::duration<double> >(afterTime - beforetime).count();
    cout<<"alltime: "<< alltime<<endl;

    return 0;
}


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sImageLeft, sImageRight;
            ss >> t;        //  时间戳
//            vTimestamps.push_back(t);
            ss >> sImageLeft;       //  左图路径
            vstrImageLeft.push_back(sImageLeft);
            ss >> sImageRight;      // 右图路径
            vstrImageRight.push_back(sImageRight);

        }
    }
}


void LoadImages_new(const string &strAssociationFilename, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string sImageLeft, sImageRight;
            double sTimestamps;
            ss >> sImageLeft;       //  左图路径
            vstrImageLeft.push_back(sImageLeft);
            ss >> sImageRight;      // 右图路径
            vstrImageRight.push_back(sImageRight);
            ss >> sTimestamps;
//            cout<<setw(6)<<setfill('0')<<sTimestamps<<endl;
            vTimestamps.push_back(sTimestamps);

        }
    }
}



