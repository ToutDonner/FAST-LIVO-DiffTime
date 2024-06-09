// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <Python.h>
#include <math.h>
#include <omp.h>
#include <ros/ros.h>
#include <so3_math.h>
#include <unistd.h>

#include <Eigen/Core>
#include <csignal>
#include <fstream>
#include <mutex>
#include <thread>
// #include <common_lib.h>
#include <cv_bridge/cv_bridge.h>
#include <fast_livo/States.h>
#include <geometry_msgs/Vector3.h>
#include <ikd-Tree/ikd_Tree.h>
#include <image_transport/image_transport.h>
#include <ivox3d/ivox3d.h>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <vikit/camera_loader.h>
#include <visualization_msgs/Marker.h>

#include <opencv2/opencv.hpp>

#include "IMU_Processing.h"
#include "lidar_selection.h"
#include "preprocess.h"

#define INIT_TIME (0.5)
#define MAXN (360000)
#define PUBFRAME_PERIOD (20)

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

/**
 * @brief ivox
 *
 */
using namespace faster_lio;

using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using VV4F =
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>;
using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

// #define USE_iVox
IVoxType::Options ivox_options_;
std::shared_ptr<IVoxType> ivox_ = nullptr;
int nearby_type;
double resolution;
bool firstPoint = true;

mutex mtx_buffer;
condition_variable sig_buffer;

// mutex mtx_buffer_pointcloud;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, img_topic, config_file;
M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);
// Vector3d Lidar_offset_to_IMU(0.04165, 0.02326, -0.0284); // Avia
Vector3d Lidar_offset_to_IMU;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0,
    laserCloudValidNum = 0, effct_feat_num = 0, time_log_counter = 0,
    publish_count = 0;
int MIN_IMG_COUNT = 0;

double res_mean_last = 0.05;
// double gyr_cov_scale, acc_cov_scale;
double gyr_cov_scale = 0, acc_cov_scale = 0;
// double last_timestamp_lidar, last_timestamp_imu = -1.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0,
       last_timestamp_img = -1.0;
// double filter_size_corner_min, filter_size_surf_min, filter_size_map_min,
// fov_deg;
double filter_size_corner_min = 0, filter_size_surf_min = 0,
       filter_size_map_min = 0, fov_deg = 0;
// double cube_len, HALF_FOV_COS, FOV_DEG, total_distance, lidar_end_time,
// first_lidar_time = 0.0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0,
       lidar_end_time = 0, first_lidar_time = 0.0;
double first_img_time = -1.0;
// double kdtree_incremental_time, kdtree_search_time;
double kdtree_incremental_time = 0, kdtree_search_time = 0,
       kdtree_delete_time = 0.0;
int kdtree_search_counter = 0, kdtree_size_st = 0, kdtree_size_end = 0,
    add_point_size = 0, kdtree_delete_counter = 0;
;
// double copy_time, readd_time, fov_check_time, readd_box_time,
// delete_box_time;
double copy_time = 0, readd_time = 0, fov_check_time = 0, readd_box_time = 0,
       delete_box_time = 0;
double T1[MAXN], T2[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN],
    s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN];

double match_time = 0, solve_time = 0, solve_const_H_time = 0;

bool lidar_pushed, flg_reset, flg_exit = false;
bool ncc_en;
int dense_map_en = 1;
int img_en = 1;
int lidar_en = 1;
int debug = 0;
int frame_num = 0;
bool fast_lio_is_ready = false;
int grid_size, patch_size;
double outlier_threshold, ncc_thre;
/**
 * @brief add by crz
 *
 */
bool onlyUpdateBias, useVio, onlyUpdateBg, useKalmanSmooth, zero_point_one;
int eigenValueThreshold;
double img_time_offset;
PointCloudXYZI::Ptr pcl_wait_test(new PointCloudXYZI());

vector<BoxPointType> cub_needrm;
vector<BoxPointType> cub_needad;
// deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<double> time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<cv::Mat> img_buffer;
deque<double> img_time_buffer;
vector<bool> point_selected_surf;
vector<vector<int>> pointSearchInd_surf;
vector<PointVector> Nearest_Points;
vector<double> res_last;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
vector<double> cameraextrinT(3, 0.0);
vector<double> cameraextrinR(9, 0.0);
double total_residual;
double LASER_POINT_COV, IMG_POINT_COV, cam_fx, cam_fy, cam_cx, cam_cy;
bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
bool lio_first = false;
// surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr cube_points_add(new PointCloudXYZI());
PointCloudXYZI::Ptr map_cur_frame_point(new PointCloudXYZI());
PointCloudXYZI::Ptr sub_map_cur_frame_point(new PointCloudXYZI());

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
Eigen::Matrix3d Rcl;
Eigen::Vector3d Pcl;

// estimator inputs and output;
LidarMeasureGroup LidarMeasures;
// SparseMap sparse_map;
StatesGroup state;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

void SigHandle(int sig) {
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp) {
  V3D rot_ang(Log(state.rot_end));
  fprintf(fp, "%lf ", LidarMeasures.lidar_beg_time - first_lidar_time);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));  // Angle
  fprintf(fp, "%lf %lf %lf ", state.pos_end(0), state.pos_end(1),
          state.pos_end(2));                   // Pos
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // omega
  fprintf(fp, "%lf %lf %lf ", state.vel_end(0), state.vel_end(1),
          state.vel_end(2));                   // Vel
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);  // Acc
  fprintf(fp, "%lf %lf %lf ", state.bias_g(0), state.bias_g(1),
          state.bias_g(2));  // Bias_g
  fprintf(fp, "%lf %lf %lf ", state.bias_a(0), state.bias_a(1),
          state.bias_a(2));  // Bias_a
  fprintf(fp, "%lf %lf %lf ", state.gravity(0), state.gravity(1),
          state.gravity(2));  // Bias_a
  fprintf(fp, "\r\n");
  fflush(fp);
}

void pointBodyToWorld(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) {
  V3D p_body(pi[0], pi[1], pi[2]);

  V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state.rot_end * (p_body + Lidar_offset_to_IMU) + state.pos_end);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;

  float intensity = pi->intensity;
  intensity = intensity - floor(intensity);

  int reflection_map = intensity * 10000;
}

#ifndef USE_ikdforest
int points_cache_size = 0;
void points_cache_collect() {
  PointVector points_history;
  ikdtree.acquire_removed_points(points_history);
  points_cache_size = points_history.size();
}
#endif

BoxPointType get_cube_point(float center_x, float center_y, float center_z) {
  BoxPointType cube_points;
  V3F center_p(center_x, center_y, center_z);
  // cout<<"center_p: "<<center_p.transpose()<<endl;

  for (int i = 0; i < 3; i++) {
    cube_points.vertex_max[i] = center_p[i] + 0.5 * cube_len;
    cube_points.vertex_min[i] = center_p[i] - 0.5 * cube_len;
  }

  return cube_points;
}

BoxPointType get_cube_point(float xmin, float ymin, float zmin, float xmax,
                            float ymax, float zmax) {
  BoxPointType cube_points;
  cube_points.vertex_max[0] = xmax;
  cube_points.vertex_max[1] = ymax;
  cube_points.vertex_max[2] = zmax;
  cube_points.vertex_min[0] = xmin;
  cube_points.vertex_min[1] = ymin;
  cube_points.vertex_min[2] = zmin;
  return cube_points;
}

#ifndef USE_ikdforest
BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment() {
  cub_needrm.clear();
  kdtree_delete_counter = 0;
  kdtree_delete_time = 0.0;
  pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
  V3D pos_LiD = state.pos_end;

  if (!Localmap_Initialized) {
    // if (cube_len <= 2.0 * MOV_THRESHOLD * DET_RANGE) throw
    // std::invalid_argument("[Error]: Local Map Size is too small! Please
    // change parameter \"cube_side_length\" to larger than %d in the launch
    // file.\n");
    for (int i = 0; i < 3; i++) {
      LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
      LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
    }
    Localmap_Initialized = true;
    return;
  }
  // printf("Local Map is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n",
  // LocalMap_Points.vertex_min[0],LocalMap_Points.vertex_max[0],LocalMap_Points.vertex_min[1],LocalMap_Points.vertex_max[1],LocalMap_Points.vertex_min[2],LocalMap_Points.vertex_max[2]);
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
        dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
      need_move = true;
  }
  if (!need_move) return;
  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points;
  float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                       double(DET_RANGE * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = LocalMap_Points;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
      // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n",
      // tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
      // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n",
      // tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
    }
  }
  LocalMap_Points = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm.size() > 0)
    kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
  kdtree_delete_time = omp_get_wtime() - delete_begin;
  // printf("Delete time: %0.6f, delete size:
  // %d\n",kdtree_delete_time,kdtree_delete_counter); printf("Delete Box:
  // %d\n",int(cub_needrm.size()));
}
#endif

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  mtx_buffer.lock();
  // cout<<"got feature"<<endl;
  sensor_msgs::PointCloud2::Ptr msg_in(new sensor_msgs::PointCloud2(*msg));
  if (zero_point_one) {
    msg_in->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - 0.1);
  }
  if (msg_in->header.stamp.toSec() < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg_in, ptr);
  // ROS_INFO("get point cloud at time: %.6f and size: %d",
  // msg_in->header.stamp.toSec() - 0.1, ptr->points.size());
  printf("[ INFO ]: get point cloud at time: %.6f and size: %d.\n",
         msg_in->header.stamp.toSec(), int(ptr->points.size()));
  lidar_buffer.push_back(ptr);

  time_buffer.push_back(msg_in->header.stamp.toSec());
  last_timestamp_lidar = msg_in->header.stamp.toSec();

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  mtx_buffer.lock();
  if (msg->header.stamp.toSec() < last_timestamp_lidar) {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  printf("[ INFO ]: get point cloud at time: %.6f.\n",
         msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(msg->header.stamp.toSec());
  last_timestamp_lidar = msg->header.stamp.toSec();

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  publish_count++;
  // cout<<"msg_in:"<<msg_in->header.stamp.toSec()<<endl;
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  double timestamp = msg->header.stamp.toSec();
  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    ROS_ERROR("imu loop back, clear buffer");
    imu_buffer.clear();
    flg_reset = true;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
  cv::Mat img;
  /**
   * @brief 这个问题已经被修复了，老版本代码是toCvShare
   *        不知道为啥图像会挂掉，
   *
   */
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}
/**
 * @brief 在图像回调的时候减去曝光时间和传输时间
 *        网口理论最大传输速度125Mb/s 1万个像素大概0.2289ms
 *        如果是USB自己换算一下
 *
 * @param msg
 */
void img_cbk(const sensor_msgs::ImageConstPtr &msg) {
  // cout<<"In Img_cbk"<<endl;
  // if (first_img_time<0 && time_buffer.size()>0) {
  //     first_img_time = msg->header.stamp.toSec() - time_buffer.front();
  // }-
  if (!img_en) {
    return;
  }
  ROS_INFO("get img at time: %.6f",
           msg->header.stamp.toSec() - img_time_offset);
  if (msg->header.stamp.toSec() - img_time_offset < last_timestamp_img) {
    ROS_ERROR("img loop back, clear buffer");
    img_buffer.clear();
    img_time_buffer.clear();
  }
  mtx_buffer.lock();
  // cout<<"Lidar_buff.size()"<<lidar_buffer.size()<<endl;
  // cout<<"Imu_buffer.size()"<<imu_buffer.size()<<endl;
  img_buffer.push_back(getImageFromMsg(msg));
  img_time_buffer.push_back(msg->header.stamp.toSec() - img_time_offset);
  last_timestamp_img = msg->header.stamp.toSec() - img_time_offset;
  // cv::imshow("img", img);
  // cv::waitKey(1);
  // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

bool sync_packages(LidarMeasureGroup &meas) {
  if ((lidar_buffer.empty() &&
       img_buffer.empty())) {  // has lidar topic or img topic?
    return false;
  }
  // ROS_ERROR("In sync");
  if (meas.is_lidar_end)  // If meas.is_lidar_end==true, means it just after
                          // scan end, clear all buffer in meas.
  {
    meas.measures.clear();
    meas.is_lidar_end = false;
  }

  if (!lidar_pushed) {  // If not in lidar scan, need to generate new meas
    if (lidar_buffer.empty()) {
      // ROS_ERROR("out sync");
      return false;
    }
    meas.lidar = lidar_buffer.front();  // push the firsrt lidar topic
    if (meas.lidar->points.size() <= 1) {
      mtx_buffer.lock();
      if (img_buffer.size() >
          0)  // temp method, ignore img topic when no lidar points, keep sync
      {
        lidar_buffer.pop_front();
        img_buffer.pop_front();
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();
      // ROS_ERROR("out sync");
      return false;
    }
    sort(meas.lidar->points.begin(), meas.lidar->points.end(),
         time_list);                            // sort by sample timestamp
    meas.lidar_beg_time = time_buffer.front();  // generate lidar_beg_time
    lidar_end_time =
        meas.lidar_beg_time + meas.lidar->points.back().curvature /
                                  double(1000);  // calc lidar scan end time
    lidar_pushed = true;                         // flag
  }

  if (img_buffer.empty()) {  // no img topic, means only has lidar topic
    if (last_timestamp_imu <
        lidar_end_time + 0.02) {  // imu message needs to be larger than
                                  // lidar_end_time, keep complete propagate.
      // ROS_ERROR("out sync");
      return false;
    }
    struct MeasureGroup m;  // standard method to keep imu message.
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    m.imu.clear();
    mtx_buffer.lock();
    while ((!imu_buffer.empty() && (imu_time < lidar_end_time))) {
      imu_time = imu_buffer.front()->header.stamp.toSec();
      if (imu_time > lidar_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false;  // sync one whole lidar scan.
    meas.is_lidar_end =
        true;  // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_ERROR("out sync");
    return true;
  }
  struct MeasureGroup m;
  // cout<<"lidar_buffer.size(): "<<lidar_buffer.size()<<" img_buffer.size():
  // "<<img_buffer.size()<<endl; cout<<"time_buffer.size():
  // "<<time_buffer.size()<<" img_time_buffer.size():
  // "<<img_time_buffer.size()<<endl; cout<<"img_time_buffer.front():
  // "<<img_time_buffer.front()<<"lidar_end_time:
  // "<<lidar_end_time<<"last_timestamp_imu: "<<last_timestamp_imu<<endl;
  if ((img_time_buffer.front() >
       lidar_end_time)) {  // has img topic, but img topic timestamp larger than
                           // lidar end time, process lidar topic.
    if (last_timestamp_imu < lidar_end_time + 0.02) {
      // ROS_ERROR("out sync");
      return false;
    }
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    m.imu.clear();
    mtx_buffer.lock();
    //这里把lidar end time之前的IMU数据push进入m.imu
    while ((!imu_buffer.empty() && (imu_time < lidar_end_time))) {
      imu_time = imu_buffer.front()->header.stamp.toSec();
      if (imu_time > lidar_end_time) break;
      if (imu_time < meas.lidar_beg_time) {
        imu_buffer.pop_front();
        continue;
      }
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false;
    meas.is_lidar_end = true;
    meas.measures.push_back(m);
  } else {
    double img_start_time =
        img_time_buffer.front();  // process img topic, record timestamp
    if (last_timestamp_imu < img_start_time) {
      // ROS_ERROR("out sync");
      return false;
    }
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    m.imu.clear();
    m.img_offset_time =
        img_start_time -
        meas.lidar_beg_time;  // record img offset time, it shoule be the Kalman
                              // update timestamp.
    cout << "[debug]------" << m.img_offset_time << endl;
    m.img = img_buffer.front();
    m.img_rcv_time = img_time_buffer.front();
    mtx_buffer.lock();
    /**
     * @brief 原版代码相机时间与雷达时间是完全一致的，
     *        如果使用时间不一致的数据应该会导致雷达去畸变所使用的imu被提前pop
     * front
     *
     * @param imu_buffer
     */
    for (const auto &it : imu_buffer) {
      imu_time = it->header.stamp.toSec();
      if (imu_time > img_start_time) break;
      m.imu.push_back(it);
    }

    img_buffer.pop_front();
    img_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    meas.is_lidar_end =
        false;  // has img topic in lidar scan, so flag "is_lidar_end=false"
    meas.measures.push_back(m);
  }
  // ROS_ERROR("out sync");
  return true;
}

void map_incremental() {
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    pointBodyToWorld(&(feats_down_body->points[i]),
                     &(feats_down_world->points[i]));
  }
#ifndef USE_iVox
  ikdtree.Add_Points(feats_down_world->points, true);
#else
  ivox_->AddPoints(feats_down_world->points);
#endif
}

// PointCloudXYZRGB::Ptr pcl_wait_pub_RGB(new PointCloudXYZRGB(500000, 1));
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
void publish_frame_world_rgb(const ros::Publisher &pubLaserCloudFullRes,
                             lidar_selection::LidarSelectorPtr lidar_selector) {
  uint size = pcl_wait_pub->points.size();
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
  if (img_en) {
    laserCloudWorldRGB->clear();
    for (int i = 0; i < size; i++) {
      PointTypeRGB pointRGB;
      pointRGB.x = pcl_wait_pub->points[i].x;
      pointRGB.y = pcl_wait_pub->points[i].y;
      pointRGB.z = pcl_wait_pub->points[i].z;
      V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y,
              pcl_wait_pub->points[i].z);
      V2D pc(lidar_selector->new_frame_->w2c(p_w));
      if (lidar_selector->new_frame_->cam_->isInFrame(pc.cast<int>(), 0)) {
        // cv::Mat img_cur = lidar_selector->new_frame_->img();
        cv::Mat img_rgb = lidar_selector->img_rgb;
        V3F pixel = lidar_selector->getpixel(img_rgb, pc);
        pointRGB.r = pixel[2];
        pointRGB.g = pixel[1];
        pointRGB.b = pixel[0];
        laserCloudWorldRGB->push_back(pointRGB);
      }
    }
  }

  if (1)  // if(publish_count >= PUBFRAME_PERIOD)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    if (img_en) {
      // cout<<"RGB pointcloud size: "<<laserCloudWorldRGB->size()<<endl;
      pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
    } else {
      pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
    }
    laserCloudmsg.header.stamp =
        ros::Time::now();  //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
    // pcl_wait_pub->clear();
  }
  // mtx_buffer_pointcloud.unlock();
}

// PointCloudXYZI::Ptr pcl_wait_test(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_test_World(new PointCloudXYZI());
/**
 * @brief 发布转换后的点云，时间添加时间补偿时使用
 *
 * @param pubLaserCloudFullRes
 * @param lidar_selector
 */
void publish_frame_wrold_rgb2(
    const ros::Publisher &pubLaserCloudFullRes,
    lidar_selection::LidarSelectorPtr lidar_selector) {
  uint size = pcl_wait_test_World->points.size();
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
  if (img_en) {
    laserCloudWorldRGB->clear();
    for (int i = 0; i < size; i++) {
      PointTypeRGB pointRGB;
      pointRGB.x = pcl_wait_test_World->points[i].x;
      pointRGB.y = pcl_wait_test_World->points[i].y;
      pointRGB.z = pcl_wait_test_World->points[i].z;
      V3D p_w(pcl_wait_test_World->points[i].x,
              pcl_wait_test_World->points[i].y,
              pcl_wait_test_World->points[i].z);
      V2D pc(lidar_selector->new_frame_->w2c(p_w));
      if (lidar_selector->new_frame_->cam_->isInFrame(pc.cast<int>(), 0)) {
        // cv::Mat img_cur = lidar_selector->new_frame_->img();
        cv::Mat img_rgb = lidar_selector->img_rgb;
        V3F pixel = lidar_selector->getpixel(img_rgb, pc);
        pointRGB.r = pixel[2];
        pointRGB.g = pixel[1];
        pointRGB.b = pixel[0];
        laserCloudWorldRGB->push_back(pointRGB);
      }
    }
  }
  // else
  // {
  //*pcl_wait_pub = *laserCloudWorld;
  // }
  // mtx_buffer_pointcloud.lock();
  if (1)  // if(publish_count >= PUBFRAME_PERIOD)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    if (img_en) {
      // cout<<"RGB pointcloud size: "<<laserCloudWorldRGB->size()<<endl;
      pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
    } else {
      pcl::toROSMsg(*pcl_wait_test_World, laserCloudmsg);
    }
    laserCloudmsg.header.stamp =
        ros::Time::now();  //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
    pcl_wait_test_World->clear();
  }
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes) {
  // PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort :
  // feats_down_body); int size = laserCloudFullRes->points.size(); if(size==0)
  // return; PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

  // for (int i = 0; i < size; i++)
  // {
  //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
  // }
  uint size = pcl_wait_pub->points.size();
  // PointCloudXYZ::Ptr laserCloudWorld(new PointCloudXYZ(size, 1));
  // else
  // {
  //*pcl_wait_pub = *laserCloudWorld;
  // }
  // mtx_buffer_pointcloud.lock();
  if (1)  // if(publish_count >= PUBFRAME_PERIOD)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);

    laserCloudmsg.header.stamp =
        ros::Time::now();  //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
    // pcl_wait_pub->clear();
  }
  // mtx_buffer_pointcloud.unlock();
}

void publish_frame_world2(const ros::Publisher &pubLaserCloudFullRes) {
  // PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort :
  // feats_down_body); int size = laserCloudFullRes->points.size(); if(size==0)
  // return; PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

  // for (int i = 0; i < size; i++)
  // {
  //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
  // }
  uint size = pcl_wait_test_World->points.size();
  // PointCloudXYZ::Ptr laserCloudWorld(new PointCloudXYZ(size, 1));
  // else
  // {
  //*pcl_wait_pub = *laserCloudWorld;
  // }
  // mtx_buffer_pointcloud.lock();
  if (1)  // if(publish_count >= PUBFRAME_PERIOD)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*pcl_wait_test_World, laserCloudmsg);

    laserCloudmsg.header.stamp =
        ros::Time::now();  //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
    // pcl_wait_pub->clear();
  }
  // mtx_buffer_pointcloud.unlock();
}
void map_incremental2() {
#ifndef USE_iVox
  ikdtree.Add_Points(pcl_wait_test_World->points, true);
#else
  ivox_->AddPoints(pcl_wait_test_World->points);
#endif
}
void publish_visual_world_map(const ros::Publisher &pubVisualCloud) {
  PointCloudXYZI::Ptr laserCloudFullRes(map_cur_frame_point);
  int size = laserCloudFullRes->points.size();
  if (size == 0) return;
  // PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

  // for (int i = 0; i < size; i++)
  // {
  //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
  // }
  // mtx_buffer_pointcloud.lock();
  PointCloudXYZI::Ptr pcl_visual_wait_pub(new PointCloudXYZI());
  *pcl_visual_wait_pub = *laserCloudFullRes;
  if (1)  // if(publish_count >= PUBFRAME_PERIOD)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_visual_wait_pub, laserCloudmsg);
    laserCloudmsg.header.stamp =
        ros::Time::now();  //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubVisualCloud.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
    // pcl_wait_pub->clear();
  }
  // mtx_buffer_pointcloud.unlock();
}

void publish_visual_world_sub_map(const ros::Publisher &pubSubVisualCloud) {
  PointCloudXYZI::Ptr laserCloudFullRes(sub_map_cur_frame_point);
  int size = laserCloudFullRes->points.size();
  if (size == 0) return;
  // PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

  // for (int i = 0; i < size; i++)
  // {
  //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
  // }
  // mtx_buffer_pointcloud.lock();
  PointCloudXYZI::Ptr sub_pcl_visual_wait_pub(new PointCloudXYZI());
  *sub_pcl_visual_wait_pub = *laserCloudFullRes;
  if (1)  // if(publish_count >= PUBFRAME_PERIOD)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_wait_pub, laserCloudmsg);
    laserCloudmsg.header.stamp =
        ros::Time::now();  //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualCloud.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
    // pcl_wait_pub->clear();
  }
  // mtx_buffer_pointcloud.unlock();
}

void publish_effect_world(const ros::Publisher &pubLaserCloudEffect) {
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
  for (int i = 0; i < effct_feat_num; i++) {
    RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp =
      ros::Time::now();  //.fromSec(last_timestamp_lidar);
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher &pubLaserCloudMap) {
  sensor_msgs::PointCloud2 laserCloudMap;
  pcl::toROSMsg(*featsFromMap, laserCloudMap);
  laserCloudMap.header.stamp = ros::Time::now();
  laserCloudMap.header.frame_id = "camera_init";
  pubLaserCloudMap.publish(laserCloudMap);
}

template <typename T>
void set_posestamp(T &out) {
#ifdef USE_IKFOM
  // state_ikfom stamp_state = kf.get_x();
  out.position.x = state_point.pos(0);
  out.position.y = state_point.pos(1);
  out.position.z = state_point.pos(2);
#else
  out.position.x = state.pos_end(0);
  out.position.y = state.pos_end(1);
  out.position.z = state.pos_end(2);
#endif
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped) {
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time().fromSec(last_timestamp_lidar);
  // //.ros::Time()fromSec(last_timestamp_lidar);
  // odomAftMapped.header.stamp =
  //     ros::Time::now();  //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);
  // odomAftMapped.twist.twist.linear.x = state_point.vel(0);
  // odomAftMapped.twist.twist.linear.y = state_point.vel(1);
  // odomAftMapped.twist.twist.linear.z = state_point.vel(2);
  // if (Measures.imu.size()>0) {
  //     Vector3d tmp(Measures.imu.back()->angular_velocity.x,
  //     Measures.imu.back()->angular_velocity.y,Measures.imu.back()->angular_velocity.z);
  //     odomAftMapped.twist.twist.angular.x = tmp[0] - state_point.bg(0);
  //     odomAftMapped.twist.twist.angular.y = tmp[1] - state_point.bg(1);
  //     odomAftMapped.twist.twist.angular.z = tmp[2] - state_point.bg(2);
  // }
  // static tf::TransformBroadcaster br;
  // tf::Transform                   transform;
  // tf::Quaternion                  q;
  // transform.setOrigin(tf::Vector3(state.pos_end(0), state.pos_end(1),
  // state.pos_end(2))); q.setW(geoQuat.w); q.setX(geoQuat.x);
  // q.setY(geoQuat.y);
  // q.setZ(geoQuat.z);
  // transform.setRotation( q );
  // br.sendTransform( tf::StampedTransform( transform,
  // odomAftMapped.header.stamp, "camera_init", "aft_mapped" ) );
  pubOdomAftMapped.publish(odomAftMapped);
}

void publish_mavros(const ros::Publisher &mavros_pose_publisher) {
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_odom_frame";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void publish_path(const ros::Publisher pubPath) {
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}

/**
 * @brief 发布雷达坐标系下的点云
 *
 * @param pubLaserCloudFrame
 */
void publish_frame(const ros::Publisher &pubLaserCloudFrame) {
  uint size = pcl_wait_pub->points.size();
  // PointCloudXYZ::Ptr laserCloudWorld(new PointCloudXYZ(size, 1));
  // else
  // {
  //*pcl_wait_pub = *laserCloudWorld;
  // }
  // mtx_buffer_pointcloud.lock();
  if (1)  // if(publish_count >= PUBFRAME_PERIOD)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;

    pcl::toROSMsg(*pcl_wait_test, laserCloudmsg);

    // laserCloudmsg.header.stamp = ros::Time().fromSec(last_timestamp_lidar);
    // //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.stamp =
        ros::Time::now();  //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFrame.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
    // pcl_wait_pub->clear();
  }
}

void readParameters(ros::NodeHandle &nh) {
  nh.param<int>("dense_map_enable", dense_map_en, 1);
  nh.param<int>("img_enable", img_en, 1);
  nh.param<int>("lidar_enable", lidar_en, 1);
  nh.param<int>("debug", debug, 0);
  nh.param<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
  nh.param<bool>("ncc_en", ncc_en, false);
  nh.param<int>("min_img_count", MIN_IMG_COUNT, 1000);
  nh.param<double>("cam_fx", cam_fx, 453.483063);
  nh.param<double>("cam_fy", cam_fy, 453.254913);
  nh.param<double>("cam_cx", cam_cx, 318.908851);
  nh.param<double>("cam_cy", cam_cy, 234.238189);
  nh.param<double>("laser_point_cov", LASER_POINT_COV, 0.001);
  nh.param<double>("img_point_cov", IMG_POINT_COV, 10);
  nh.param<string>("map_file_path", map_file_path, "");
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<string>("camera/img_topic", img_topic, "/usb_cam/image_raw");
  nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
  nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<double>("filter_size_map", filter_size_map_min, 0.5);
  nh.param<double>("cube_side_length", cube_len, 200);
  nh.param<double>("mapping/fov_degree", fov_deg, 180);
  nh.param<double>("mapping/gyr_cov_scale", gyr_cov_scale, 1.0);
  nh.param<double>("mapping/acc_cov_scale", acc_cov_scale, 1.0);
  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
  nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
  nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, 0);
  nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
  nh.param<vector<double>>("camera/Pcl", cameraextrinT, vector<double>());
  nh.param<vector<double>>("camera/Rcl", cameraextrinR, vector<double>());
  nh.param<int>("grid_size", grid_size, 40);
  nh.param<int>("patch_size", patch_size, 4);
  nh.param<double>("outlier_threshold", outlier_threshold, 100);
  nh.param<double>("ncc_thre", ncc_thre, 100);
  /**
   * @brief add by crz
   *
   */
  nh.param<bool>("onlyUpdateBias", onlyUpdateBias, false);
  nh.param<bool>("onlyUpdateBg", onlyUpdateBg, false);
  nh.param<bool>("useKalmanSmooth", useKalmanSmooth, true);
  nh.param<bool>("zero_point_one", zero_point_one, false);
  nh.param<double>("img_time_offset", img_time_offset, 0.0);
  nh.param<bool>("useVio", useVio, true);
  nh.param<int>("eigenValueThreshold", eigenValueThreshold, 0);
  nh.param<int>("nearby_type", nearby_type, 18);
  nh.param<double>("resolution", resolution, 0.45);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "laserMapping");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  readParameters(nh);
  cout << "debug:" << debug << " MIN_IMG_COUNT: " << MIN_IMG_COUNT << endl;
  pcl_wait_pub->clear();
  // pcl_visual_wait_pub->clear();
  ros::Subscriber sub_pcl =
      p_pre->lidar_type == AVIA
          ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk)
          : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
  ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
  ros::Subscriber sub_img = nh.subscribe(img_topic, 200000, img_cbk);
  image_transport::Publisher img_pub = it.advertise("/rgb_img", 1);
  ros::Publisher pubLaserCloudFullRes =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  ros::Publisher pubLaserCloudFullResRgb =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_rgb", 100);
  ros::Publisher pubVisualCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_map", 100);
  ros::Publisher pubSubVisualCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map", 100);
  ros::Publisher pubLaserCloudEffect =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
  ros::Publisher pubLaserCloudMap =
      nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
  ros::Publisher pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
  /* add by crz*/
  ros::Publisher pubLaserCloudFrame =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100);

#ifdef DEPLOY
  ros::Publisher mavros_pose_publisher =
      nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
#endif

  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";

  /*** variables definition ***/
  VD(DIM_STATE)
  solution;
  MD(DIM_STATE, DIM_STATE)
  G, H_T_H, I_STATE, G_k;
  V3D rot_add, t_add;
  StatesGroup state_propagat;
  StatesGroup state_last_lidar;

  PointType pointOri, pointSel, coeff;

  // PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
  int effect_feat_num = 0, frame_num = 0;
  double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0,
                         aver_time_match = 0, aver_time_solve = 0,
                         aver_time_const_H_time = 0;

  FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
  HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                 filter_size_surf_min);
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min,
                                filter_size_map_min);

  shared_ptr<ImuProcess> p_imu(new ImuProcess());
  // p_imu->set_extrinsic(V3D(0.04165, 0.02326, -0.0284));   //avia
  // p_imu->set_extrinsic(V3D(0.05512, 0.02226, -0.0297));   //horizon
  V3D extT;
  M3D extR;
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);
  Lidar_offset_to_IMU = extT;
  lidar_selection::LidarSelectorPtr lidar_selector(
      new lidar_selection::LidarSelector(grid_size, new SparseMap));
  if (!vk::camera_loader::loadFromRosNs("laserMapping", lidar_selector->cam))
    throw std::runtime_error("Camera model not correctly specified.");
  lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;
  lidar_selector->debug = debug;
  lidar_selector->patch_size = patch_size;
  lidar_selector->outlier_threshold = outlier_threshold;
  lidar_selector->ncc_thre = ncc_thre;
  lidar_selector->sparse_map->set_camera2lidar(cameraextrinR, cameraextrinT);
  lidar_selector->set_extrinsic(extT, extR);
  lidar_selector->state = &state;
  lidar_selector->state_propagat = &state_propagat;
  lidar_selector->NUM_MAX_ITERATIONS = NUM_MAX_ITERATIONS;
  lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT;
  lidar_selector->img_point_cov = IMG_POINT_COV;
  lidar_selector->fx = cam_fx;
  lidar_selector->fy = cam_fy;
  lidar_selector->cx = cam_cx;
  lidar_selector->cy = cam_cy;
  lidar_selector->ncc_en = ncc_en;
  /* add by crz */
  lidar_selector->eigenValueThreshold = eigenValueThreshold;
  lidar_selector->init();

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
  p_imu->set_acc_cov_scale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
  p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
  p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));
  p_imu->set_state_last_lidar(state_last_lidar);
  p_imu->set_G_k(G_k);
  // p_imu->G_k = &G_k;
  // p_imu->state_last_lidar = &state_last_lidar;
  /**
   * @brief ivox
   *
   */
  ivox_options_.resolution_ = resolution;
  if (nearby_type == 0) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
  } else if (nearby_type == 6) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
  } else if (nearby_type == 18) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  } else if (nearby_type == 26) {
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
  } else {
    LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
    ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
  }
  ivox_ = std::make_shared<IVoxType>(ivox_options_);

  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();
  G_k.setZero();

  /*** debug record ***/
  FILE *fp;
  string pos_log_dir = root_dir + "/Log/pos_log.txt";
  fp = fopen(pos_log_dir.c_str(), "w");

  ofstream fout_pre, fout_out, fout_dbg, fout_pose;
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
  fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);
  fout_pose.open(root_dir + "/image/pose.txt", ios::out);
  // if (fout_pre && fout_out)
  //     cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
  // else
  //     cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

  //------------------------------------------------------------------------------------------------------
  signal(SIGINT, SigHandle);
  ros::Rate rate(5000);
  bool status = ros::ok();
  while (status) {
    if (flg_exit) break;
    ros::spinOnce();
    if (!sync_packages(LidarMeasures)) {
      status = ros::ok();
      cv::waitKey(1);
      rate.sleep();
      continue;
    }

    /*** Packaged got ***/
    if (flg_reset) {
      ROS_WARN("reset when rosbag play back");
      p_imu->Reset();
      flg_reset = false;
      continue;
    }

    // double t0,t1,t2,t3,t4,t5,match_start, match_time, solve_start,
    // solve_time, svd_time;
    double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;
    double time0, time1, time2, time3, time4, time5, ivox_add, serach_time;

    match_time = kdtree_search_time = kdtree_search_counter = solve_time =
        solve_const_H_time = svd_time = 0;
    t0 = omp_get_wtime();

    p_imu->Process2(LidarMeasures, state, feats_undistort);
    state_propagat = state;

    if (lidar_selector->debug) {
      LidarMeasures.debug_show();
    }

    if (feats_undistort->empty() || (feats_undistort == nullptr)) {
      // cout<<" No point!!!"<<endl;
      if (!fast_lio_is_ready) {
        first_lidar_time = LidarMeasures.lidar_beg_time;
        p_imu->first_lidar_time = first_lidar_time;
        LidarMeasures.measures.clear();
        cout << "FAST-LIO not ready" << endl;
        continue;
      }
    } else {
      int size = feats_undistort->points.size();
    }
    fast_lio_is_ready = true;
    flg_EKF_inited =
        (LidarMeasures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false
                                                                      : true;
    /**
     * @brief useVio 为false时可以单独测试LIO
     *
     */
    if (!LidarMeasures.is_lidar_end && useVio && fast_lio_is_ready &&
        lio_first) {
      cout << "[ VIO ]: Raw feature num: " << pcl_wait_test->points.size()
           << "." << endl;
      if (first_lidar_time < 10) {
        continue;
      }
      // cout<<"cur state:"<<state.rot_end<<endl;
      if (img_en) {
        euler_cur = RotMtoEuler(state.rot_end);
        fout_pre << setw(20)
                 << LidarMeasures.last_update_time - first_lidar_time << " "
                 << euler_cur.transpose() * 57.3 << " "
                 << state.pos_end.transpose() << " "
                 << state.vel_end.transpose() << " " << state.bias_g.transpose()
                 << " " << state.bias_a.transpose() << " "
                 << state.gravity.transpose() << endl;

        // lidar_selector->detect(LidarMeasures.measures.back().img,
        // feats_undistort); mtx_buffer_pointcloud.lock();

        // int size = feats_undistort->points.size();
        // cout<<"size1111111111111111: "<<size<<endl;
        // PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
        // for (int i = 0; i < size; i++)
        // {
        //     pointBodyToWorld(&feats_undistort->points[i], \
                //                         &laserCloudWorld->points[i]);
        // }
        int ptsize = pcl_wait_test->points.size();
        // pcl_wait_test_World(new PointCloudXYZI(ptsize, 1));
        pcl_wait_test_World->resize(ptsize);
        for (int i = 0; i < ptsize; i++) {
          pointBodyToWorld(&pcl_wait_test->points[i],
                           &pcl_wait_test_World->points[i]);
        }

        cv::Mat img_copy = LidarMeasures.measures.back().img.clone();

        lidar_selector->detect(LidarMeasures.measures.back().img,
                               pcl_wait_test_World);
        // int size = lidar_selector->map_cur_frame_.size();
        int size_sub = lidar_selector->sub_map_cur_frame_.size();

        // map_cur_frame_point->clear();
        sub_map_cur_frame_point->clear();
        // for(int i=0; i<size; i++)
        // {
        //     PointType temp_map;
        //     temp_map.x = lidar_selector->map_cur_frame_[i]->pos_[0];
        //     temp_map.y = lidar_selector->map_cur_frame_[i]->pos_[1];
        //     temp_map.z = lidar_selector->map_cur_frame_[i]->pos_[2];
        //     temp_map.intensity = 0.;
        //     map_cur_frame_point->push_back(temp_map);
        // }
        for (int i = 0; i < size_sub; i++) {
          PointType temp_map;
          temp_map.x = lidar_selector->sub_map_cur_frame_[i]->pos_[0];
          temp_map.y = lidar_selector->sub_map_cur_frame_[i]->pos_[1];
          temp_map.z = lidar_selector->sub_map_cur_frame_[i]->pos_[2];
          temp_map.intensity = 0.;
          sub_map_cur_frame_point->push_back(temp_map);
        }
        cv::Mat img_rgb = lidar_selector->img_cp;
        cv_bridge::CvImage out_msg;
        out_msg.header.stamp = ros::Time::now();
        // out_msg.header.frame_id = "camera_init";
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = img_rgb;
        img_pub.publish(out_msg.toImageMsg());

        /**
         * @brief 如果存在时间补偿，使用第二个赋色。
         *
         */
        publish_frame_world_rgb(pubLaserCloudFullResRgb, lidar_selector);
        // if (img_time_offset == 0) {
        // } else {
        //   publish_frame_wrold_rgb2(pubLaserCloudFullResRgb, lidar_selector);
        // }
        // publish_visual_world_sub_map(pubSubVisualCloud);

        // *map_cur_frame_point = *pcl_wait_pub;
        // mtx_buffer_pointcloud.unlock();
        // lidar_selector->detect(LidarMeasures.measures.back().img,
        // feats_down_world);
        // p_imu->push_update_state(LidarMeasures.measures.back().img_offset_time,
        // state); geoQuat =
        // tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1),
        // euler_cur(2));

        euler_cur = RotMtoEuler(state.rot_end);
        fout_out << setw(20)
                 << LidarMeasures.last_update_time - first_lidar_time << " "
                 << euler_cur.transpose() * 57.3 << " "
                 << state.pos_end.transpose() << " "
                 << state.vel_end.transpose() << " " << state.bias_g.transpose()
                 << " " << state.bias_a.transpose() << " "
                 << state.gravity.transpose() << " "
                 << feats_undistort->points.size() << endl;

        /**
         * @brief 如果img和lidar时间不完全一致，VIO只更新Bias
         *        理论有待验证，反正能跑.....
         *
         */
        if (onlyUpdateBias) {
          state_last_lidar += G_k * (state - state_propagat);
          // state_last_lidar += G_k * (state_propagat - state);
          state_last_lidar.cov +=
              G_k * (state.cov - state_propagat.cov) * G_k.transpose();
          // state_last_lidar.cov += G_k * (state_propagat.cov - state.cov) *
          // G_k.transpose();
          state = state_last_lidar;
          /* debug */
          // cout << "[DEBUG]------------->VIO" << state.pos_end << endl;
          // cout << "[DEBUG]------------->VIO" << state_last_lidar.pos_end <<
          // endl;
        }
        if (useKalmanSmooth) {
          // compute F_k from x_lidar to x_camera_hat
          MD(DIM_STATE, 1)
          F_lc_18x1 = state_propagat - state_last_lidar;

          MD(DIM_STATE, DIM_STATE)
          F_lc = Matrix<double, 18, 18>::Zero();
          F_lc(0, 0) = F_lc_18x1(0, 0);
          F_lc(1, 1) = F_lc_18x1(1, 0);
          F_lc(2, 2) = F_lc_18x1(2, 0);
          F_lc(3, 3) = F_lc_18x1(3, 0);
          F_lc(4, 4) = F_lc_18x1(4, 0);
          F_lc(5, 5) = F_lc_18x1(5, 0);
          F_lc(6, 6) = F_lc_18x1(6, 0);
          F_lc(7, 7) = F_lc_18x1(7, 0);
          F_lc(8, 8) = F_lc_18x1(8, 0);
          F_lc(9, 9) = F_lc_18x1(9, 0);
          F_lc(10, 10) = F_lc_18x1(10, 0);
          F_lc(11, 11) = F_lc_18x1(11, 0);
          F_lc(12, 12) = F_lc_18x1(12, 0);
          F_lc(13, 13) = F_lc_18x1(13, 0);
          F_lc(14, 14) = F_lc_18x1(14, 0);
          F_lc(15, 15) = F_lc_18x1(15, 0);
          F_lc(16, 16) = F_lc_18x1(16, 0);
          F_lc(17, 17) = F_lc_18x1(17, 0);

          MD(DIM_STATE, DIM_STATE)
          G_lc = state_last_lidar.cov * F_lc.transpose() *
                 state_propagat.cov.inverse();
          state_last_lidar = state_last_lidar + G_lc * (state - state_propagat);
          state_last_lidar.cov =
              state_last_lidar.cov +
              G_lc * (state.cov - state_propagat.cov) * G_lc.transpose();

          // save data
          // frame_num += 1;
          // string img_name = std::to_string(frame_num) + ".png";
          // cv::imwrite(root_dir + "/image/" + img_name, img_copy);
          // Eigen::Quaterniond rot_(state.rot_end);

          // fout_pose << frame_num << " " << rot_.w() << " " << rot_.x() << " "
          //           << rot_.y() << " " << rot_.z() << " "
          //           << state.pos_end.transpose() << " " << int(1) << " "
          //           << img_name << " " << endl;
          /**
           * @brief 关掉了视觉里程计
           *
           */
          // publish_odometry(pubOdomAftMapped);
        }
      }

      continue;
    }
/*** Segment the map in lidar FOV ***/
#ifndef USE_iVox
    lasermap_fov_segment();
#endif
    /*** downsample the feature points in a scan ***/
    downSizeFilterSurf.setInputCloud(feats_undistort);
    downSizeFilterSurf.filter(*feats_down_body);

    /*** initialize the map kdtree ***/
#ifndef USE_iVox

    if (ikdtree.Root_Node == nullptr) {
      if (feats_down_body->points.size() > 5) {
        ikdtree.set_downsample_param(filter_size_map_min);
        ikdtree.Build(feats_down_body->points);
      }
      continue;
    }
    int featsFromMapNum = ikdtree.size();

#else
    //  ivox //

    if (firstPoint) {
      ivox_->AddPoints(feats_down_body->points);
      firstPoint = false;
    }
#endif
    //  ivox //
    // ROS_WARN("222222");
    feats_down_size = feats_down_body->points.size();
    // cout << "[ LIO ]: Raw feature num: " << feats_undistort->points.size() <<
    // " downsamp num " << feats_down_size << " Map num: " << featsFromMapNum <<
    // "." << endl;
    cout << "[ LIO ]: Raw feature num: " << feats_undistort->points.size()
         << " downsamp num " << feats_down_size << "." << endl;

    /*** ICP and iterated Kalman filter update ***/
    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);
    // vector<double> res_last(feats_down_size, 1000.0); // initial //
    res_last.resize(feats_down_size, 1000.0);

    t1 = omp_get_wtime();
    if (lidar_en) {
      euler_cur = RotMtoEuler(state.rot_end);
      fout_pre << setw(20) << LidarMeasures.last_update_time - first_lidar_time
               << " " << euler_cur.transpose() * 57.3 << " "
               << state.pos_end.transpose() << " " << state.vel_end.transpose()
               << " " << state.bias_g.transpose() << " "
               << state.bias_a.transpose() << " " << state.gravity.transpose()
               << endl;
    }

    point_selected_surf.resize(feats_down_size, true);
    pointSearchInd_surf.resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);
    int rematch_num = 0;
    bool nearest_search_en = true;  //

    t2 = omp_get_wtime();

/*** iterated state estimation ***/
#ifdef MP_EN
    printf("[ LIO ]: Using multi-processor, used core number: %d.\n",
           MP_PROC_NUM);
#endif
    double t_update_start = omp_get_wtime();

    if (lidar_en) {
      for (iterCount = -1; iterCount < NUM_MAX_ITERATIONS && flg_EKF_inited;
           iterCount++) {
        match_start = omp_get_wtime();
        PointCloudXYZI().swap(*laserCloudOri);
        PointCloudXYZI().swap(*corr_normvect);
        // laserCloudOri->clear();
        // corr_normvect->clear();
        total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
        // omp_set_num_threads(MP_PROC_NUM);
        omp_set_num_threads(6);
#pragma omp parallel for
#endif
        // normvec->resize(feats_down_size);
        for (int i = 0; i < feats_down_size; i++) {
          PointType &point_body = feats_down_body->points[i];
          PointType &point_world = feats_down_world->points[i];
          V3D p_body(point_body.x, point_body.y, point_body.z);
          /* transform to world frame */
          pointBodyToWorld(&point_body, &point_world);
          vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
          auto &points_near = Nearest_Points[i];

          uint8_t search_flag = 0;
          double search_start = omp_get_wtime();
          if (nearest_search_en) {
/** Find the closest surfaces in the map **/
#ifndef USE_iVox
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near,
                                   pointSearchSqDis);
            point_selected_surf[i] =
                pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
#else

            ivox_->GetClosestPoint(point_world, points_near, NUM_MATCH_POINTS);
            point_selected_surf[i] = points_near.size() >= 3;
#endif
            kdtree_search_time += omp_get_wtime() - search_start;
            // cout << "[search_start]" << search_start << endl;
            kdtree_search_counter++;
          }

          // if (!point_selected_surf[i]) continue;

          // Debug
          // if (points_near.size()<5) {
          //     printf("\nERROR: Return Points is less than 5\n\n");
          //     printf("Target Point is:
          //     (%0.3f,%0.3f,%0.3f)\n",point_world.x,point_world.y,point_world.z);
          // }
          if (!point_selected_surf[i] || points_near.size() < NUM_MATCH_POINTS)
            continue;

          VF(4)
          pabcd;
          point_selected_surf[i] = false;
          if (esti_plane(pabcd, points_near, 0.1f))  //(planeValid)
          {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                        pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9) {
              point_selected_surf[i] = true;
              normvec->points[i].x = pabcd(0);
              normvec->points[i].y = pabcd(1);
              normvec->points[i].z = pabcd(2);
              normvec->points[i].intensity = pd2;
              res_last[i] = abs(pd2);
            }
          }
        }
        // cout<<"pca time test: "<<pca_time1<<" "<<pca_time2<<endl;
        effct_feat_num = 0;
        laserCloudOri->resize(feats_down_size);
        corr_normvect->reserve(feats_down_size);
        for (int i = 0; i < feats_down_size; i++) {
          if (point_selected_surf[i] && (res_last[i] <= 2.0)) {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num++;
          }
        }

        res_mean_last = total_residual / effct_feat_num;
        // cout << "[ mapping ]: Effective feature num: "<<effct_feat_num<<"
        // res_mean_last "<<res_mean_last<<endl;
        match_time += omp_get_wtime() - match_start;
        solve_start = omp_get_wtime();

        /*** Computation of Measuremnt Jacobian matrix H and measurents vector
         * ***/
        MatrixXd Hsub(effct_feat_num, 6);
        VectorXd meas_vec(effct_feat_num);

        for (int i = 0; i < effct_feat_num; i++) {
          const PointType &laser_p = laserCloudOri->points[i];
          V3D point_this(laser_p.x, laser_p.y, laser_p.z);
          point_this += Lidar_offset_to_IMU;
          M3D point_crossmat;
          point_crossmat << SKEW_SYM_MATRX(point_this);

          /*** get the normal vector of closest surface/corner ***/
          const PointType &norm_p = corr_normvect->points[i];
          V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

          /*** calculate the Measuremnt Jacobian matrix H ***/
          V3D A(point_crossmat * state.rot_end.transpose() * norm_vec);
          Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;

          /*** Measuremnt: distance to the closest surface/corner ***/
          meas_vec(i) = -norm_p.intensity;
        }
        solve_const_H_time += omp_get_wtime() - solve_start;

        MatrixXd K(DIM_STATE, effct_feat_num);

        EKF_stop_flg = false;
        flg_EKF_converged = false;

        /*** Iterative Kalman Filter Update ***/
        if (!flg_EKF_inited) {
          cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
          /*** only run in initialization period ***/
          MatrixXd H_init(MD(9, DIM_STATE)::Zero());
          MatrixXd z_init(VD(9)::Zero());
          H_init.block<3, 3>(0, 0) = M3D::Identity();
          H_init.block<3, 3>(3, 3) = M3D::Identity();
          H_init.block<3, 3>(6, 15) = M3D::Identity();
          z_init.block<3, 1>(0, 0) = -Log(state.rot_end);
          z_init.block<3, 1>(0, 0) = -state.pos_end;

          auto H_init_T = H_init.transpose();
          auto &&K_init =
              state.cov * H_init_T *
              (H_init * state.cov * H_init_T + 0.0001 * MD(9, 9)::Identity())
                  .inverse();
          solution = K_init * z_init;

          // solution.block<9,1>(0,0).setZero();
          // state += solution;
          // state.cov = (MatrixXd::Identity(DIM_STATE, DIM_STATE) - K_init *
          // H_init) * state.cov;

          state.resetpose();
          EKF_stop_flg = true;
        } else {
          auto &&Hsub_T = Hsub.transpose();
          auto &&HTz = Hsub_T * meas_vec;
          H_T_H.block<6, 6>(0, 0) = Hsub_T * Hsub;
          /**
           * @brief 特征值分解，看一下过曝时H矩阵特征值是否有变化 0327
           *
           */
          // EigenSolver<Matrix<double, 6, 6>> es(H_T_H.block<6, 6>(0, 0));
          // auto V = es.eigenvalues().real();
          // cout << "[Lidar EigenValue-->]" << V.transpose() << endl;

          MD(DIM_STATE, DIM_STATE) &&K_1 =
              (H_T_H + (state.cov / LASER_POINT_COV).inverse()).inverse();
          G.block<DIM_STATE, 6>(0, 0) =
              K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
          auto vec = state_propagat - state;
          solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec -
                     G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);

          int minRow, minCol;
          if (0)  // if(V.minCoeff(&minRow, &minCol) < 1.0f)
          {
            VD(6)
            V = H_T_H.block<6, 6>(0, 0).eigenvalues().real();
            cout << "!!!!!! Degeneration Happend, eigen values: "
                 << V.transpose() << endl;
            EKF_stop_flg = true;
            solution.block<6, 1>(9, 0).setZero();
          }

          state += solution;

          rot_add = solution.block<3, 1>(0, 0);
          t_add = solution.block<3, 1>(3, 0);

          if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) {
            flg_EKF_converged = true;
          }

          deltaR = rot_add.norm() * 57.3;
          deltaT = t_add.norm() * 100;
        }
        euler_cur = RotMtoEuler(state.rot_end);

        /*** Rematch Judgement ***/
        nearest_search_en = false;
        if (flg_EKF_converged ||
            ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2)))) {
          nearest_search_en = true;
          rematch_num++;
        }

        /*** Convergence Judgements and Covariance Update ***/
        if (!EKF_stop_flg &&
            (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))) {
          if (flg_EKF_inited) {
            /*** Covariance Update ***/
            // G.setZero();
            // G.block<DIM_STATE,6>(0,0) = K * Hsub;
            state.cov = (I_STATE - G) * state.cov;
            total_distance += (state.pos_end - position_last).norm();
            position_last = state.pos_end;
            geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
                euler_cur(0), euler_cur(1), euler_cur(2));

            VD(DIM_STATE)
            K_sum = K.rowwise().sum();
            VD(DIM_STATE)
            P_diag = state.cov.diagonal();
            // cout<<"K: "<<K_sum.transpose()<<endl;
            // cout<<"P: "<<P_diag.transpose()<<endl;
            // cout<<"position: "<<state.pos_end.transpose()<<" total distance:
            // "<<total_distance<<endl;
          }
          EKF_stop_flg = true;
        }
        solve_time += omp_get_wtime() - solve_start;

        /* debug */
        // cout << "[DEBUG]------------->LIO" << state_last_lidar.pos_end <<
        // endl; cout << "[DEBUG]------------->LIO" << state.pos_end << endl;
        /**
         * @brief state_last_lidar 保存LIO结束时的状态
         *
         */
        state_last_lidar = state;
        if (EKF_stop_flg) break;
      }
    }

    // cout<<"[ mapping ]: iteration count: "<<iterCount+1<<endl;
    // SaveTrajTUM(LidarMeasures.lidar_beg_time, state.rot_end, state.pos_end);
    double t_update_end = omp_get_wtime();
    /******* Publish odometry *******/
    euler_cur = RotMtoEuler(state.rot_end);
    geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
        euler_cur(0), euler_cur(1), euler_cur(2));
    publish_odometry(pubOdomAftMapped);
    lio_first = true;

    /*** add the feature points to map kdtree ***/
    // t3 = omp_get_wtime();
    time0 = omp_get_wtime();
    map_incremental();
    time1 = omp_get_wtime();

    // t5 = omp_get_wtime();
    // kdtree_incremental_time = t5 - t3 + readd_time;
    ivox_add = time1 - time0;
    cout << "[map increamental time]---------->" << ivox_add << endl;
    cout << "[nearest search time]-------------->" << kdtree_search_time << ","
         << kdtree_search_counter << endl;
    cout << "[average point search time]-------->"
         << kdtree_search_time / kdtree_search_counter << endl;
    /******* Publish points *******/

    PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort
                                                       : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                          &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub = *laserCloudWorld;
    *pcl_wait_test = *laserCloudFullRes;

    /**
     * @brief 发布imu系下点云
     *
     */
    publish_frame(pubLaserCloudFrame);

    publish_frame_world(pubLaserCloudFullRes);
    // publish_visual_world_map(pubVisualCloud);
    publish_effect_world(pubLaserCloudEffect);
    // publish_map(pubLaserCloudMap);
    publish_path(pubPath);
  }
  //--------------------------save map---------------
  // string surf_filename(map_file_path + "/surf.pcd");
  // string corner_filename(map_file_path + "/corner.pcd");
  // string all_points_filename(map_file_path + "/all_points.pcd");

  // PointCloudXYZI surf_points, corner_points;
  // surf_points = *featsFromMap;
  // fout_out.close();
  // fout_pre.close();
  // if (surf_points.size() > 0 && corner_points.size() > 0)
  // {
  // pcl::PCDWriter pcd_writer;
  // cout << "saving...";
  // pcd_writer.writeBinary(surf_filename, surf_points);
  // pcd_writer.writeBinary(corner_filename, corner_points);
  // }
#ifndef DEPLOY
  vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
  FILE *fp2;
  string log_dir = root_dir + "/Log/fast_livo_time_log.csv";
  fp2 = fopen(log_dir.c_str(), "w");
  fprintf(fp2,
          "time_stamp, average time, incremental time, search time,fov check "
          "time, total time, alpha_bal, alpha_del\n");
  for (int i = 0; i < time_log_counter; i++) {
    fprintf(fp2, "%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%f,%f\n", T1[i],
            s_plot[i], s_plot2[i], s_plot3[i], s_plot4[i], s_plot5[i],
            s_plot6[i], s_plot7[i]);
    t.push_back(T1[i]);
    s_vec.push_back(s_plot[i]);
    s_vec2.push_back(s_plot2[i]);
    s_vec3.push_back(s_plot3[i]);
    s_vec4.push_back(s_plot4[i]);
    s_vec5.push_back(s_plot5[i]);
    s_vec6.push_back(s_plot6[i]);
    s_vec7.push_back(s_plot7[i]);
  }
  fclose(fp2);
  if (!t.empty()) {
    // plt::named_plot("incremental time",t,s_vec2);
    // plt::named_plot("search_time",t,s_vec3);
    // plt::named_plot("total time",t,s_vec5);
    // plt::named_plot("average time",t,s_vec);
    // plt::legend();
    // plt::show();
    // plt::pause(0.5);
    // plt::close();
  }
  cout << "no points saved" << endl;
#endif

  return 0;
}
