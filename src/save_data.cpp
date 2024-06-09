#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "saveData");
  ros::NodeHandle nh;

  ros::Subscriber sub_img;
  ros::Subscriber sub_odom;
  ros::Subscriber sub_pc;

  return 0;
}
