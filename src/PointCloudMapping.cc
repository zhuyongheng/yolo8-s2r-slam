/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "PointCloudMapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "Converter.h"
#include "System.h"

#include <pcl/filters/passthrough.h>

namespace ORB_SLAM3
{
// int currentloopcount = 0;
PointCloudMapping::PointCloudMapping(double resolution_, double meank_, double thresh_)
    : mabIsUpdating(false)
{
    this->resolution = resolution_;
    this->meank = meank_;
    this->thresh = thresh_;
    std::cout<<resolution<<" "<<meank<<" "<<thresh<<std::endl;
    statistical_filter = new pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA>(true);
    voxel = new pcl::VoxelGrid<pcl::PointXYZRGBA>();
    statistical_filter->setMeanK(meank);
    statistical_filter->setStddevMulThresh(thresh);
//    voxel->setLeafSize(resolution, resolution, resolution);
    voxel->setLeafSize(1.0f, 1.0f, 1.0f);
    globalMap = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>);

    viewerThread = make_shared<thread>(bind(&PointCloudMapping::viewer, this));
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::Clear()
{
    std::cout << "清除稠密地图" << std::endl;
    std::unique_lock<std::mutex> lck(mMutexGlobalMap);
    globalMap.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
}

void PointCloudMapping::insertKeyFrame(KeyFrame *kf)
{
    // cout << "receive a keyframe, 第" << kf->mnId << "个" << endl;
    if (kf->imLeftRgb.empty())
        return;
    unique_lock<mutex> lck(keyframeMutex);
    mlNewKeyFrames.emplace_back(kf);  //给PointCloudMapping下的mlNewKeyFrames赋值，也就是插入当前地图Atlas下的关键帧列表
    if(mlNewKeyFrames.size() > 35)
        mlNewKeyFrames.pop_front();

}

//通过传入的当前地图Atlas储存关键帧的列表生成彩色点云地图  mlNewKeyFrameForDenseMap->mlNewKeyFrames->lNewKeyFrames
void PointCloudMapping::generatePointCloud(KeyFrame *kf) //,Eigen::Isometry3d T
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    // point cloud is null ptr
    for (int m = 0; m < kf->imDepth.rows; m += 3)
    {
        for (int n = 0; n < kf->imDepth.cols; n += 3)
        {
            float d = kf->imDepth.ptr<float>(m)[n];
//            if (d < 0.01|| d > 6)
            if (d < 0.01 || d > 9)
                continue;
            pcl::PointXYZRGBA p;
            p.z = d;
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy;

            p.b = kf->imLeftRgb.ptr<uchar>(m)[n * 3];
            p.g = kf->imLeftRgb.ptr<uchar>(m)[n * 3 + 1];
            p.r = kf->imLeftRgb.ptr<uchar>(m)[n * 3 + 2];

            pPointCloud->points.push_back(p);
        }
    }
    pPointCloud->height = 1;
    pPointCloud->width = pPointCloud->points.size();
    pPointCloud->is_dense = true;
    kf->mptrPointCloud = pPointCloud;
}

void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    //////
//    //声明octomap变量
//    cout<<"copy data into octomap..."<<endl;
//    // 创建带颜色的八叉树对象，参数为分辨率，这里设成了0.05
//    octomap::ColorOcTree tree( 0.03 );

    //////


    while (1)
    {
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag)
            {
                break;
            }
        }
        if (bStop || mabIsUpdating)
        {
            continue;
        }
//        int N;
        std::list<KeyFrame *> lNewKeyFrames;
        {
            unique_lock<mutex> lck(keyframeMutex);
//            N = mlNewKeyFrames.size();
            lNewKeyFrames = mlNewKeyFrames;
//            if(N == 0)
//                continue;
//            else
//            {
//                mlNewKeyFrames.clear();
//            }
        }

//        Clear();
        //初始化
        for (auto pKF : lNewKeyFrames)
        {
            if (pKF->isBad())
                continue;

//            std::chrono::steady_clock::time_point t11 = std::chrono::steady_clock::now();
            generatePointCloud(pKF);
//            std::chrono::steady_clock::time_point t22 = std::chrono::steady_clock::now();
//            double constructMap= std::chrono::duration_cast<std::chrono::duration<double> >(t22 - t11).count();
//            cout<<"constructMap: "<<constructMap<<endl;

            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p(new pcl::PointCloud<pcl::PointXYZRGBA>);
            pcl::transformPointCloud(*(pKF->mptrPointCloud), *(p), Converter::toMatrix4d(pKF->GetPoseInverse()));
            {
                std::unique_lock<std::mutex> lck(mMutexGlobalMap);
                *globalMap += *p;
            }
        }



//        for (auto m:globalMap->points)
//        {
//            // 将点云里的点插入到octomap中
//            tree.updateNode( octomap::point3d(m.x, m.y, m.z), true );
//        }
//
//        // 设置颜色
//        for (auto m:globalMap->points)
//        {
//            tree.integrateNodeColor( m.x, m.y, m.z, m.r, m.g, m.b );
//        }
//        // 更新octomap
//        tree.updateInnerOccupancy();
//        // 存储octomap, 注意要存成.ot文件而非.bt文件
//        tree.write( "octomap.ot" );



//        pcl::io::savePCDFileBinary("result.pcd", *globalMap);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGBA>);
        voxel->setInputCloud(globalMap);
        voxel->setLeafSize(0.1f,0.1f,0.1f);
//        voxel->setLeafSize(0.2f,0.2f,0.2f);
        voxel->filter(*globalMap);
//        viewer.showCloud(globalMap);  // 这个比较费时，建议不需要实时显示的可以屏蔽或改成几次显示一次



    }
}

// 保存地图的函数，需要的自行调用~
void PointCloudMapping::save()
{
    std::unique_lock<std::mutex> lck(mMutexGlobalMap);
    pcl::io::savePCDFileBinary("result.pcd", *globalMap);
    cout << "globalMap save finished" << endl;
}

void PointCloudMapping::updatecloud(Map &curMap)
    {
        std::unique_lock<std::mutex> lck(updateMutex);

        mabIsUpdating = true;
        currentvpKFs = curMap.GetAllKeyFrames();
        cout<<"currentvpKFs.size:"<<currentvpKFs.size()<<endl;
        // loopbusy = true;
        cout << "开始点云更新" << endl;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmpGlobalMap(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr curPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmpGlobalMapFilter(new pcl::PointCloud<pcl::PointXYZRGBA>());
        for (int i = 0; i < currentvpKFs.size(); i++)
        {
            if (!mabIsUpdating)
            {
                std::cout << "中断点云更新" <<std::endl;
                return;
            }
            if (!currentvpKFs[i]->isBad() && currentvpKFs[i]->mptrPointCloud)
            {

                pcl::transformPointCloud(
                        *(currentvpKFs[i]->mptrPointCloud), *(curPointCloud),
                        Converter::toMatrix4d(currentvpKFs[i]->GetPoseInverse()));
                *tmpGlobalMap += *curPointCloud;

                voxel->setInputCloud(tmpGlobalMap);
                voxel->setLeafSize(10.0f,10.0f,10.0f);
                voxel->filter(*tmpGlobalMapFilter);
                tmpGlobalMap->swap(*tmpGlobalMapFilter);
            }
        }
        cout << "点云更新完成" << endl;
        {
            std::unique_lock<std::mutex> lck(mMutexGlobalMap);
            globalMap = tmpGlobalMap;
        }
        mabIsUpdating = false;
    }

/*

void PointCloudMapping::updatecloud(Map &curMap)
{
    std::unique_lock<std::mutex> lck(updateMutex);
    
    mabIsUpdating = true;
    currentvpKFs = curMap.GetAllKeyFrames();
    // loopbusy = true;
    cout << "开始点云更新" << endl;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmpGlobalMap(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr curPointCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmpGlobalMapFilter(new pcl::PointCloud<pcl::PointXYZRGBA>);
    for(int i = 0; i < currentvpKFs.size(); i++)
    {
        if (!mabIsUpdating){
            std::cout << "中断点云更新" <<std::endl;
            return;
        }
        if(!currentvpKFs[i]->isBad() && currentvpKFs[i]->mptrPointCloud)
        {

            pcl::transformPointCloud(
                    *(currentvpKFs[i]->mptrPointCloud), *(curPointCloud),
                    Converter::toMatrix4d(currentvpKFs[i]->GetPoseInverse()));
            *tmpGlobalMap += *curPointCloud;

            voxel->setInputCloud(tmpGlobalMap);
            voxel->filter(*tmpGlobalMapFilter);
            tmpGlobalMap->swap(*tmpGlobalMapFilter);
        }
    }
    cout << "点云更新完成" << endl;
    {
        std::unique_lock<std::mutex> lck(mMutexGlobalMap);
        globalMap = tmpGlobalMap;
    }
//            continue;
//
//        generatePointCloud(currentvpKFs[i]);
//        pcl::transformPointCloud(*(currentvpKFs[i]->mptrPointCloud), *(curPointCloud),Converter::toMatrix4d(currentvpKFs[i]->GetPoseInverse()));
//        {
//            std::unique_lock<std::mutex> lck(mMutexGlobalMap);
//            *tmpGlobalMap += *curPointCloud;
//        }
//    }
//    voxel->setInputCloud(tmpGlobalMap);
//    voxel->filter(*tmpGlobalMap);
//    cout << "点云更新完成" << endl;

    mabIsUpdating = false;
}
*/
}//ORBSLAM3