#include "IMU_Processing.h"

const bool time_list(PointType &x, PointType &y) {
  return (x.curvature < y.curvature);
}

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1) {
  init_iter_num = 1;
#ifdef USE_IKFOM
  Q = process_noise_cov();
#endif
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  cov_acc_scale = V3D(1, 1, 1);
  cov_gyr_scale = V3D(1, 1, 1);
  cov_bias_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_acc = V3D(0.1, 0.1, 0.1);
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  Lid_offset_to_IMU = Zero3d;
  Lid_rot_to_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
  ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init_ = true;
  start_timestamp_ = -1;
  init_iter_num = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::push_update_state(double offs_t, StatesGroup state) {
  // V3D acc_tmp(last_acc), angvel_tmp(last_ang), vel_imu(state.vel_end),
  // pos_imu(state.pos_end); M3D R_imu(state.rot_end); angvel_tmp -=
  // state.bias_g; acc_tmp   = acc_tmp * G_m_s2 / mean_acc.norm() -
  // state.bias_a; acc_tmp  = R_imu * acc_tmp + state.gravity;
  // IMUpose.push_back(set_pose6d(offs_t, acc_tmp, angvel_tmp, vel_imu, pos_imu,
  // R_imu));
  V3D acc_tmp = acc_s_last, angvel_tmp = angvel_last, vel_imu(state.vel_end),
      pos_imu(state.pos_end);
  M3D R_imu(state.rot_end);
  IMUpose.push_back(
      set_pose6d(offs_t, acc_tmp, angvel_tmp, vel_imu, pos_imu, R_imu));
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T) {
  Lid_offset_to_IMU = T.block<3, 1>(0, 3);
  Lid_rot_to_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl) {
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot) {
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler) {
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov_scale(const V3D &scaler) {
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }
void ImuProcess::set_state_last_lidar(StatesGroup &state_last) {
  state_last_lidar = state_last;
}
void ImuProcess::set_G_k(MD(DIM_STATE, DIM_STATE) & G_k_yy) { G_k = G_k_yy; }

#ifdef USE_IKFOM
void ImuProcess::IMU_init(
    const MeasureGroup &meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N) {
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;

  if (b_first_frame_) {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    // first_lidar_time = meas.lidar_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }

  for (const auto &imu : meas.imu) {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N +
              (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) *
                  (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N +
              (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) *
                  (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(-mean_acc / mean_acc.norm() * G_m_s2);

  // state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 /
  // scale_gravity)));
  init_state.bg = mean_gyr;
  init_state.offset_T_L_I = Lid_offset_to_IMU;
  init_state.offset_R_L_I = Lid_rot_to_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P =
      kf_state.get_P() * 0.001;
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();
}
#else
void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout,
                          int &N) {
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;

  if (b_first_frame_) {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    // first_lidar_time = meas.lidar_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }

  for (const auto &imu : meas.imu) {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N +
              (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) *
                  (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N +
              (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) *
                  (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N++;
  }

  state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;

  // state_inout.rot_end = Eye3d;
  state_inout.rot_end = Exp(mean_acc.cross(V3D(0, 0, 1)));
  state_inout.bias_g = mean_gyr;

  last_imu_ = meas.imu.back();
}
#endif

#ifdef USE_IKFOM
void ImuProcess::UndistortPcl(
    const MeasureGroup &meas,
    esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
    PointCloudXYZI &pcl_out) {
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time IMUpose.push_back(
      set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  time *** / pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time =
      pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel,
                               imu_state.pos,
                               imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 *
                   (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // #ifdef DEBUG_PRINT
    fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " "
             << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    acc_avr = acc_avr * G_m_s2 / mean_acc.norm();  // - state_inout.ba;

    if (head->header.stamp.toSec() < last_lidar_end_time_) {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    } else {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last = imu_state.rot * (acc_avr - imu_state.ba);
    for (int i = 0; i < 3; i++) {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel,
                                 imu_state.pos,
                                 imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);

  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

#ifdef DEBUG_PRINT
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P = kf_state.get_P();
  cout << "[ IMU Process ]: vel " << imu_state.vel.transpose() << " pos "
       << imu_state.pos.transpose() << " ba" << imu_state.ba.transpose()
       << " bg " << imu_state.bg.transpose() << endl;
  cout << "propagated cov: " << P.diagonal().transpose() << endl;
#endif

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    acc_imu << VEC_FROM_ARRAY(tail->acc);
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
       * represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt -
               imu_state.pos);
      V3D P_compensate =
          imu_state.offset_R_L_I.conjugate() *
          (imu_state.rot.conjugate() *
               (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) +
                T_ei) -
           imu_state.offset_T_L_I);  // not accurate!

      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
#else

void ImuProcess::Forward(const MeasureGroup &meas, StatesGroup &state_inout,
                         double pcl_beg_time, double end_time) {
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();

  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end,
  // state.pos_end, state.rot_end));
  if (IMUpose.empty()) {
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last,
                                 state_inout.vel_end, state_inout.pos_end,
                                 state_inout.rot_end));
  }

  /*** forward propagation at each imu point ***/
  V3D acc_imu = acc_s_last, angvel_avr = angvel_last, acc_avr,
      vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  M3D R_imu(state_inout.rot_end);
  //  last_state = state_inout;
  MD(DIM_STATE, DIM_STATE)
  F_x, cov_w;

  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

    // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y,
    // tail->angular_velocity.z;

    acc_avr << 0.5 *
                   (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
    last_acc = acc_avr;
    last_ang = angvel_avr;
    // #ifdef DEBUG_PRINT
    fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " "
             << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    angvel_avr -= state_inout.bias_g;
    acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

    if (head->header.stamp.toSec() < last_lidar_end_time_) {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    } else {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    // cout<<setw(20)<<"dt: "<<dt<<endl;
    /* covariance propagation */
    M3D acc_avr_skew;
    M3D Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

    F_x.setIdentity();
    cov_w.setZero();

    F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
    F_x.block<3, 3>(0, 9) = -Eye3d * dt;
    // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
    F_x.block<3, 3>(3, 6) = Eye3d * dt;
    F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
    F_x.block<3, 3>(6, 12) = -R_imu * dt;
    F_x.block<3, 3>(6, 15) = Eye3d * dt;

    cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
    cov_w.block<3, 3>(6, 6) =
        R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3, 3>(9, 9).diagonal() =
        cov_bias_gyr * dt * dt;  // bias gyro covariance
    cov_w.block<3, 3>(12, 12).diagonal() =
        cov_bias_acc * dt * dt;  // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    /* propogation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propogation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(
        set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (end_time - imu_end_time);
  state_inout.vel_end = vel_imu + note * acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  state_inout.pos_end =
      pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;

  last_imu_ = v_imu.back();
  last_lidar_end_time_ = end_time;

  // auto pos_liD_e = state_inout.pos_end + state_inout.rot_end *
  // Lid_offset_to_IMU; auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

#ifdef DEBUG_PRINT
  cout << "[ IMU Process ]: vel " << state_inout.vel_end.transpose() << " pos "
       << state_inout.pos_end.transpose() << " ba"
       << state_inout.bias_a.transpose() << " bg "
       << state_inout.bias_g.transpose() << endl;
  cout << "propagated cov: " << state_inout.cov.diagonal().transpose() << endl;
#endif
}

void ImuProcess::Backward(const LidarMeasureGroup &lidar_meas,
                          StatesGroup &state_inout, PointCloudXYZI &pcl_out) {
  /*** undistort each lidar point (backward propagation) ***/
  M3D R_imu;
  V3D acc_imu, angvel_avr, vel_imu, pos_imu;
  double dt;
  auto pos_liD_e =
      state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    acc_imu << VEC_FROM_ARRAY(head->acc);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    angvel_avr << VEC_FROM_ARRAY(head->gyr);
    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
       * represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt +
               R_i * Lid_offset_to_IMU - pos_liD_e);

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
#endif

#ifdef USE_IKFOM
void ImuProcess::Process(const LidarMeasureGroup &lidar_meas,
                         esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloudXYZI::Ptr cur_pcl_un_) {
  double t1, t2, t3;
  t1 = omp_get_wtime();
  MeasureGroup meas = lidar_meas.measures.back();
  if (meas.imu.empty()) {
    return;
  };
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_) {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;

    last_imu_ = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT) {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO(
          "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f "
          "%.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f "
          "%.8f",
          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2],
          mean_acc.norm(), cov_acc_scale[0], cov_acc_scale[1], cov_acc_scale[2],
          cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1],
          cov_gyr[2]);
      cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
      cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word
      // frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO(
          "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f "
          "%.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f "
          "%.8f",
          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2],
          mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2],
          cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1],
          cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  if (lidar_meas.is_lidar_end) {
    UndistortPcl(lidar_meas, kf_state, *cur_pcl_un_);
  }

  t2 = omp_get_wtime();

  // {
  //   static ros::Publisher pub_UndistortPcl =
  //       nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
  //   sensor_msgs::PointCloud2 pcl_out_msg;
  //   pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
  //   pcl_out_msg.header.stamp = ros::Time().fromSec(meas.lidar_beg_time);
  //   pcl_out_msg.header.frame_id = "/livox";
  //   pub_UndistortPcl.publish(pcl_out_msg);
  // }

  t3 = omp_get_wtime();

  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
#else
void ImuProcess::Process(const LidarMeasureGroup &lidar_meas, StatesGroup &stat,
                         PointCloudXYZI::Ptr cur_pcl_un_) {
  double t1, t2, t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  MeasureGroup meas = lidar_meas.measures.back();

  if (imu_need_init_) {
    if (meas.imu.empty()) {
      return;
    };
    /// The very first lidar frame
    IMU_init(meas, stat, init_iter_num);

    imu_need_init_ = true;

    last_imu_ = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT) {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO(
          "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f "
          "%.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f "
          "%.8f",
          stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(),
          cov_acc_scale[0], cov_acc_scale[1], cov_acc_scale[2], cov_acc[0],
          cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
      cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);

      // cov_acc = Eye3d * cov_acc_scale;
      // cov_gyr = Eye3d * cov_gyr_scale;
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word
      // frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO(
          "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f "
          "%.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f "
          "%.8f",
          stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(),
          cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0],
          cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  if (lidar_meas.is_lidar_end) {
    /*** sort point clouds by offset time ***/
    *cur_pcl_un_ = *(lidar_meas.lidar);
    sort(cur_pcl_un_->points.begin(), cur_pcl_un_->points.end(), time_list);
    const double &pcl_beg_time = lidar_meas.lidar_beg_time;
    const double &pcl_end_time =
        pcl_beg_time + lidar_meas.lidar->points.back().curvature / double(1000);
    Forward(meas, stat, pcl_beg_time, pcl_end_time);
    // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
    //        <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;
    // cout<<"Time:";
    // for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
    //   cout<<it->offset_time<<" ";
    // }
    // cout<<endl<<"size:"<<IMUpose.size()<<endl;
    Backward(lidar_meas, stat, *cur_pcl_un_);
    last_lidar_end_time_ = pcl_end_time;
    IMUpose.clear();
  } else {
    const double &pcl_beg_time = lidar_meas.lidar_beg_time;
    const double &img_end_time = pcl_beg_time + meas.img_offset_time;
    Forward(meas, stat, pcl_beg_time, img_end_time);
  }

  t2 = omp_get_wtime();

  // {
  //   static ros::Publisher pub_UndistortPcl =
  //       nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
  //   sensor_msgs::PointCloud2 pcl_out_msg;
  //   pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
  //   pcl_out_msg.header.stamp = ros::Time().fromSec(meas.lidar_beg_time);
  //   pcl_out_msg.header.frame_id = "/livox";
  //   pub_UndistortPcl.publish(pcl_out_msg);
  // }

  t3 = omp_get_wtime();

  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}

void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas,
                              StatesGroup &state_inout,
                              PointCloudXYZI &pcl_out) {
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  MeasureGroup meas;
  meas = lidar_meas.measures.back();
  // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double pcl_beg_time =
      MAX(lidar_meas.lidar_beg_time, lidar_meas.last_update_time);
  // const double &pcl_beg_time = meas.lidar_beg_time;

  /*** sort point clouds by offset time ***/
  pcl_out.clear();
  auto pcl_it =
      lidar_meas.lidar->points.begin() + lidar_meas.lidar_scan_index_now;
  auto pcl_it_end = lidar_meas.lidar->points.end();
  const double pcl_end_time =
      lidar_meas.is_lidar_end
          ? lidar_meas.lidar_beg_time +
                lidar_meas.lidar->points.back().curvature / double(1000)
          : lidar_meas.lidar_beg_time +
                lidar_meas.measures.back().img_offset_time;
  const double pcl_offset_time =
      lidar_meas.is_lidar_end
          ? (pcl_end_time - lidar_meas.lidar_beg_time) * double(1000)
          : 0.0;
  while (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time) {
    pcl_out.push_back(*pcl_it);
    pcl_it++;
    lidar_meas.lidar_scan_index_now++;
  }
  // cout<<"pcl_offset_time:  "<<pcl_offset_time<<"pcl_it->curvature:
  // "<<pcl_it->curvature<<endl;
  // cout<<"lidar_meas.lidar_scan_index_now:"<<lidar_meas.lidar_scan_index_now<<endl;
  lidar_meas.last_update_time = pcl_end_time;
  if (lidar_meas.is_lidar_end) {
    lidar_meas.lidar_scan_index_now = 0;
  }
  // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // lidar_meas.debug_show();
  // cout<<"UndistortPcl [ IMU Process ]: Process lidar from "<<pcl_beg_time<<"
  // to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to
  //          "<<imu_end_time<<endl;
  // cout<<"v_imu.size: "<<v_imu.size()<<endl;
  /*** Initialize IMU pose ***/
  IMUpose.clear();
  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end,
  // state.pos_end, state.rot_end));
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last,
                               state_inout.vel_end, state_inout.pos_end,
                               state_inout.rot_end));

  /*** forward propagation at each imu point ***/
  V3D acc_imu(acc_s_last), angvel_avr(angvel_last), acc_avr,
      vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
  M3D R_imu(state_inout.rot_end);
  MD(DIM_STATE, DIM_STATE)
  F_x, cov_w, F_x_last, cov_w_last;

  // double dt_l_i = meas.img_rcv_time - last_lidar_end_time_;
  // {
  //   StatesGroup state_;
  //   F_x_last.setIdentity();
  //   M3D acc_avr_skew_last;
  //   acc_avr_skew_last << SKEW_SYM_MATRX(angvel_last);
  //   F_x_last.block<3, 3>(0, 0) = Exp(angvel_last, -dt_l_i);
  //   F_x_last.block<3, 3>(0, 9) = -Eye3d * dt_l_i;
  //   // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
  //   F_x_last.block<3, 3>(3, 6) = Eye3d * dt_l_i;
  //   F_x_last.block<3, 3>(6, 0) = -R_imu * acc_avr_skew_last * dt_l_i;
  //   F_x_last.block<3, 3>(6, 12) = -R_imu * dt_l_i;
  //   F_x_last.block<3, 3>(6, 15) = Eye3d * dt_l_i;

  //   cov_w_last.block<3, 3>(0, 0).diagonal() = cov_gyr * dt_l_i * dt_l_i;
  //   cov_w_last.block<3, 3>(6, 6) = R_imu * cov_acc.asDiagonal() *
  //   R_imu.transpose() * dt_l_i * dt_l_i; cov_w_last.block<3, 3>(9,
  //   9).diagonal() = cov_bias_gyr * dt_l_i * dt_l_i;   // bias gyro covariance
  //   cov_w_last.block<3, 3>(12, 12).diagonal() = cov_bias_acc * dt_l_i *
  //   dt_l_i; // bias acc covariance

  //   state_.cov = F_x_last * state_last_lidar.cov * F_x_last.transpose() +
  //   cov_w_last; G_k = state_last_lidar.cov * F_x_last.transpose() *
  //   state_.cov.inverse();
  // }

  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu != v_imu.end() - 1; it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

    // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y,
    // tail->angular_velocity.z;

    acc_avr << 0.5 *
                   (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // #ifdef DEBUG_PRINT
    fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " "
             << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    angvel_avr -= state_inout.bias_g;
    acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

    if (head->header.stamp.toSec() < last_lidar_end_time_) {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    } else {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    /* covariance propagation */
    M3D acc_avr_skew;
    M3D Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

    F_x.setIdentity();
    cov_w.setZero();

    F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
    F_x.block<3, 3>(0, 9) = -Eye3d * dt;
    // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
    F_x.block<3, 3>(3, 6) = Eye3d * dt;
    F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
    F_x.block<3, 3>(6, 12) = -R_imu * dt;
    F_x.block<3, 3>(6, 15) = Eye3d * dt;

    cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
    cov_w.block<3, 3>(6, 6) =
        R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3, 3>(9, 9).diagonal() =
        cov_bias_gyr * dt * dt;  // bias gyro covariance
    cov_w.block<3, 3>(12, 12).diagonal() =
        cov_bias_acc * dt * dt;  // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    /* propogation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propogation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec():
    // "<<tail->header.stamp.toSec()<<endl;
    IMUpose.push_back(
        set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    G_k = state_last_lidar.cov * F_x.transpose() * state_inout.cov.inverse();
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  if (imu_end_time > pcl_beg_time) {
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end =
        pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
  } else {
    double note = pcl_end_time > pcl_beg_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - pcl_beg_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end =
        pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
  }

  last_imu_ = v_imu.back();
  last_lidar_end_time_ = pcl_end_time;

  auto pos_liD_e =
      state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;
  // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

  // cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos
  // "<<state_inout.pos_end.transpose()<<"
  // ba"<<state_inout.bias_a.transpose()<<" bg
  // "<<state_inout.bias_g.transpose()<<endl; cout<<"propagated cov:
  // "<<state_inout.cov.diagonal().transpose()<<endl;

  //   cout<<"UndistortPcl Time:";
  //   for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
  //     cout<<it->offset_time<<" ";
  //   }
  //   cout<<endl<<"UndistortPcl size:"<<IMUpose.size()<<endl;
  //   cout<<"Undistorted pcl_out.size: "<<pcl_out.size()
  //          <<"lidar_meas.size: "<<lidar_meas.lidar->points.size()<<endl;
  if (pcl_out.points.size() < 1) return;
  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    acc_imu << VEC_FROM_ARRAY(head->acc);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    angvel_avr << VEC_FROM_ARRAY(head->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
       * represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt +
               R_i * Lid_offset_to_IMU - pos_liD_e);

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat,
                          PointCloudXYZI::Ptr cur_pcl_un_) {
  double t1, t2, t3;
  t1 = omp_get_wtime();
  ROS_ASSERT(lidar_meas.lidar != nullptr);
  MeasureGroup meas = lidar_meas.measures.back();

  if (imu_need_init_) {
    if (meas.imu.empty()) {
      return;
    };
    /// The very first lidar frame
    IMU_init(meas, stat, init_iter_num);

    imu_need_init_ = true;

    last_imu_ = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT) {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO(
          "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f "
          "%.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f "
          "%.8f",
          stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(),
          cov_acc_scale[0], cov_acc_scale[1], cov_acc_scale[2], cov_acc[0],
          cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      cov_acc = cov_acc.cwiseProduct(cov_acc_scale);
      cov_gyr = cov_gyr.cwiseProduct(cov_gyr_scale);

      // cov_acc = Eye3d * cov_acc_scale;
      // cov_gyr = Eye3d * cov_gyr_scale;
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word
      // frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO(
          "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f "
          "%.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f "
          "%.8f",
          stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc.norm(),
          cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0],
          cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
    }

    return;
  }
  UndistortPcl(lidar_meas, stat, *cur_pcl_un_);
}

#endif