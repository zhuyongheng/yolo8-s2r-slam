//
// Created by zhu on 2023/6/28.
//

#ifndef ORB_SLAM3_OCTOMAP_H
#define ORB_SLAM3_OCTOMAP_H

#endif //ORB_SLAM3_OCTOMAP_H

#include <iostream>
#include <assert.h>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

//octomap
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

namespace ORB_SLAM3 {

class Octomap
{
public:
    octomap::ColorOcTree tree;



};


}