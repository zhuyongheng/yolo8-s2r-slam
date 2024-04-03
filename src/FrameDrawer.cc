/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>
int timestamp = 0;
namespace ORB_SLAM3
{

FrameDrawer::FrameDrawer(Atlas* pAtlas):both(false),mpAtlas(pAtlas)
{
    mState=Tracking::SYSTEM_NOT_READY;
    // 初始化图像显示画布
    // 包括：图像、特征点连线形成的轨迹（初始化时）、框（跟踪时的MapPoint）、圈（跟踪时的特征点）
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mImRight = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame(bool bOldFeatures)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    vector<pair<cv::Point2f, cv::Point2f> > vTracks;
    int state; // Tracking state

    Frame currentFrame;
    vector<MapPoint*> vpLocalMap;
    vector<cv::KeyPoint> vMatchesKeys;
    vector<MapPoint*> vpMatchedMPs;
    vector<cv::KeyPoint> vOutlierKeys;
    vector<MapPoint*> vpOutlierMPs;
    map<long unsigned int, cv::Point2f> mProjectPoints;
    map<long unsigned int, cv::Point2f> mMatchedInImage;

    //Copy variables within scoped mutex
    // step 1：将成员变量赋值给局部变量（包括图像、状态、其它的提示）
    //NOTICE 加互斥锁，避免与FrameDrawer::Update函数中图像拷贝发生冲突
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        //NOTICE 这里使用copyTo进行深拷贝是因为后面会把单通道灰度图像转为3通道图像
        mIm.copyTo(im);

        //没有初始化的时候
        if(mState==Tracking::NOT_INITIALIZED)
        {
            //获取当前帧参考帧的特征点,并且得到他们的匹配关系
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
            vTracks = mvTracks;
        }
        else if(mState==Tracking::OK)
        {
            //获取当前帧参考帧的特征点,并且得到他们的匹配关系
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;

            currentFrame = mCurrentFrame;
            vpLocalMap = mvpLocalMap;
            vMatchesKeys = mvMatchedKeys;
            vpMatchedMPs = mvpMatchedMPs;
            vOutlierKeys = mvOutlierKeys;
            vpOutlierMPs = mvpOutlierMPs;
            mProjectPoints = mmProjectPoints;
            mMatchedInImage = mmMatchedInImage;

        }
        else if(mState==Tracking::LOST)
        {
            //跟丢的时候就之获得当前帧的特征点就可以了
            vCurrentKeys = mvCurrentKeys;
        }
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    // step 2：绘制初始化轨迹连线，绘制特征点边框（特征点用小框圈住）
    // step 2.1：初始化时，当前帧的特征坐标与初始帧的特征点坐标连成线，形成轨迹
    if(state==Tracking::NOT_INITIALIZED)
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            //绘制当前帧特征点到下一帧特征点的连线,其实就是匹配关系
            //NOTICE 就是当初看到的初始化过程中图像中显示的绿线
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }
        for(vector<pair<cv::Point2f, cv::Point2f> >::iterator it=vTracks.begin(); it!=vTracks.end(); it++)
            cv::line(im,(*it).first,(*it).second, cv::Scalar(0,255,0),5);

    }
    else if(state==Tracking::OK && bOldFeatures) //TRACKING
    {
        /////*********画特征点*********////
        //当前帧追踪到的特征点计数
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                //在特征点附近正方形选择四个点
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    // 通道顺序为bgr，地图中MapPoints用绿色圆点表示，并用绿色小方框圈住cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }

        }
    }
    else if(state==Tracking::OK && !bOldFeatures)
    {
        mnTracked=0;
        int nTracked2 = 0;
        mnTrackedVO=0;
        int n = vCurrentKeys.size();

        for(int i=0; i < n; ++i)
        {

            // This is a match to a MapPoint in the map
            if(vbMap[i])
            {
                mnTracked++;
            }
        }

        map<long unsigned int, cv::Point2f>::iterator it_match = mMatchedInImage.begin();
        while(it_match != mMatchedInImage.end())
        {
            long unsigned int mp_id = it_match->first;
            cv::Point2f p_image = it_match->second;

            if(mProjectPoints.find(mp_id) != mProjectPoints.end())
            {
                cv::Point2f p_proj = mMatchedInImage[mp_id];
                cv::line(im, p_proj, p_image, cv::Scalar(0, 255, 0), 2);
                nTracked2++;
            }
            else
            {
                cv::circle(im,p_image,2,cv::Scalar(0,0,255),-1);
            }


            it_match++;
        }

        n = vOutlierKeys.size();
        for(int i=0; i < n; ++i)
        {
            cv::Point2f point3d_proy;
            float u, v;
            currentFrame.ProjectPointDistort(vpOutlierMPs[i] , point3d_proy, u, v);

            cv::Point2f point_im = vOutlierKeys[i].pt;

            cv::line(im,cv::Point2f(u, v), point_im,cv::Scalar(0, 0, 255), 1);
        }

    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    //将最终slam运行的界面保存
//    cv::imwrite("rgb_img/"+to_string(timestamp)+".png", im);
    timestamp++;

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawRightFrame()
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mImRight.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeysRight;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                         cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = mvCurrentKeysRight.size();
        const int Nleft = mvCurrentKeys.size();

        for(int i=0;i<n;i++)
        {
            if(vbVO[i + Nleft] || vbMap[i + Nleft])
            {
                cv::Point2f pt1,pt2;
                pt1.x=mvCurrentKeysRight[i].pt.x-r;
                pt1.y=mvCurrentKeysRight[i].pt.y-r;
                pt2.x=mvCurrentKeysRight[i].pt.x+r;
                pt2.y=mvCurrentKeysRight[i].pt.y+r;

                // This is a match to a MapPoint in the map
                if(vbMap[i + Nleft])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,mvCurrentKeysRight[i].pt,2,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,mvCurrentKeysRight[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}



void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nMaps = mpAtlas->CountMaps();
        int nKFs = mpAtlas->KeyFramesInMap();
        int nMPs = mpAtlas->MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    //拷贝跟踪线程的图像
//    pTracker->mImGray.copyTo(mIm);
    pTracker->mImRGB.copyTo(mIm);
    //拷贝跟踪线程的特征点
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;

    //如果上一帧的时候,追踪器没有进行初始化
    if(both){
        mvCurrentKeysRight = pTracker->mCurrentFrame.mvKeysRight;
        pTracker->mImRight.copyTo(mImRight);
        N = mvCurrentKeys.size() + mvCurrentKeysRight.size();
    }
    else{
        N = mvCurrentKeys.size();
    }

    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    //Variables for the new visualization
    mCurrentFrame = pTracker->mCurrentFrame;
    mmProjectPoints = mCurrentFrame.mmProjectPoints;
    mmMatchedInImage.clear();

    mvpLocalMap = pTracker->GetLocalMapMPS();
    mvMatchedKeys.clear();
    mvMatchedKeys.reserve(N);
    mvpMatchedMPs.clear();
    mvpMatchedMPs.reserve(N);
    mvOutlierKeys.clear();
    mvOutlierKeys.reserve(N);
    mvpOutlierMPs.clear();
    mvpOutlierMPs.reserve(N);

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;

                    mmMatchedInImage[pMP->mnId] = mvCurrentKeys[i].pt;

                }
                else
                {
                    mvpOutlierMPs.push_back(pMP);
                    mvOutlierKeys.push_back(mvCurrentKeys[i]);
                }
            }
        }

    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM
