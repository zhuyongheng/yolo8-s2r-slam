//
// Created by zhu on 2023/2/23.
//

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<string> vstrImageDepth;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageLeft, vstrImageRight, vstrImageDepth, vTimestamps);

    int nImages  = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::STEREO,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    // 主循环
    cv::Mat imLeft, imRight, imD;
    double tReadImage, tTrack;
    for(int ni=0; ni<nImages; ni++)
    {
        //计时1
//        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();


        // Read left and right images from file
        imLeft = cv::imread(string(argv[3])+"/"+vstrImageLeft[ni],cv::IMREAD_UNCHANGED);
        imRight = cv::imread(string(argv[3])+"/"+vstrImageRight[ni],cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageDepth[ni],cv::IMREAD_UNCHANGED);

//        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//
//        tReadImage = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//
//        cout<< tReadImage<<endl;

        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif


        //计时2
//        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        // Pass the images to the SLAM system
        SLAM.TrackStereo_RGBD(imLeft, imRight, imD, tframe);

//        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
//
//        tTrack = std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
//
//        cout<< tTrack<<endl;


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
//        double T=0;
//        if(ni<nImages-1)
//            T = vTimestamps[ni+1]-tframe;       //如果当前图像不是最后一帧
//        else if(ni>0)
//            T = tframe-vTimestamps[ni-1];       //如果是最后一帧

//        if(ttrack<T)
//            usleep((T-ttrack)*1e6);
            usleep(100000);


    }

    // Stop all threads
    SLAM.save();
    SLAM.Shutdown();

/***
    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
***/

    return 0;

}


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
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
            string sImageLeft, sImageRight, sD;
            ss >> t;        //  时间戳
            vTimestamps.push_back(t);
            ss >> sImageLeft;       //  左图路径
            vstrImageLeft.push_back(sImageLeft);
            ss >> sImageRight;      // 右图路径
            vstrImageRight.push_back(sImageRight);
            ss >> t;        //  深度图时间戳，在这里其实是不需要的
            ss >> sD;       //  深度图路径
            vstrImageFilenamesD.push_back(sD);

        }

    }


}






