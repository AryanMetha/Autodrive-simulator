#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <optional>
#include <algorithm>
#include <numeric>

/**
 * Extended Kalman Filter for IMU-based position estimation
 * State vector: [px, py, vx, vy, ax, ay] (position, velocity, acceleration bias)
 * This helps compensate for IMU drift and bias
 */
class IMUKalmanFilter {
public:
    IMUKalmanFilter() {
        // Initialize state vector: [px, py, vx, vy, ax_bias, ay_bias]
        state_ = Eigen::VectorXd::Zero(6);
        
        // Initialize covariance matrix (uncertainty in our estimates)
        P_ = Eigen::MatrixXd::Identity(6, 6);
        P_(0,0) = 0.001; P_(1,1) = 0.001;  // Position uncertainty
        P_(2,2) = 0.01;  P_(3,3) = 0.01;   // Velocity uncertainty  
        P_(4,4) = 0.1;   P_(5,5) = 0.1;    // Acceleration bias uncertainty
        
        // Process noise covariance (how much we trust the model)
        Q_ = Eigen::MatrixXd::Identity(6, 6);
        Q_(0,0) = 0.001; Q_(1,1) = 0.001;  // Position process noise
        Q_(2,2) = 0.01;  Q_(3,3) = 0.01;   // Velocity process noise
        Q_(4,4) = 0.001; Q_(5,5) = 0.001;  // Bias process noise (small - biases change slowly)
        
        // Measurement noise covariance for IMU accelerations
        R_imu_ = Eigen::MatrixXd::Identity(2, 2);
        R_imu_(0,0) = 0.05; R_imu_(1,1) = 0.05;  // IMU acceleration noise
        
        // Measurement noise covariance for speed measurements
        R_speed_ = Eigen::MatrixXd::Identity(1, 1);
        R_speed_(0,0) = 0.01;  // Speed sensor noise
        
        dt_ = 0.01;  // 100Hz update rate
        initialized_ = false;
    }
    
    void initialize(double x, double y) {
        state_(0) = x;
        state_(1) = y;
        state_(2) = 0.0;  // Initial velocity x
        state_(3) = 0.0;  // Initial velocity y
        state_(4) = 0.0;  // Initial acceleration bias x
        state_(5) = 0.0;  // Initial acceleration bias y
        initialized_ = true;
    }
    
    /**
     * Prediction step: Use kinematic model to predict next state
     * Position update: p = p + v*dt + 0.5*a*dt^2
     * Velocity update: v = v + a*dt
     * Bias remains constant in prediction
     */
    void predict(double ax_world, double ay_world) {
        if (!initialized_) return;
        
        // State transition matrix
        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(6, 6);
        F(0,2) = dt_;  // px depends on vx
        F(1,3) = dt_;  // py depends on vy
        
        // Control input matrix (for accelerations)
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(6, 2);
        B(2,0) = dt_;  // vx affected by ax
        B(3,1) = dt_;  // vy affected by ay
        B(0,0) = 0.5 * dt_ * dt_;  // px affected by ax
        B(1,1) = 0.5 * dt_ * dt_;  // py affected by ay
        
        // Control input (corrected accelerations)
        Eigen::Vector2d u;
        u << ax_world - state_(4), ay_world - state_(5);  // Remove bias
        
        // Predict state
        state_ = F * state_ + B * u;
        
        // Predict covariance
        P_ = F * P_ * F.transpose() + Q_;
    }
    
    /**
     * Update step with IMU acceleration measurements
     * Corrects the predicted state based on acceleration observations
     */
    void updateWithIMU(double ax_measured, double ay_measured) {
        if (!initialized_) return;
        
        // Measurement matrix (we observe accelerations)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 6);
        H(0,4) = -1.0;  // ax_observed = ax_true - ax_bias
        H(1,5) = -1.0;  // ay_observed = ay_true - ay_bias
        
        // Innovation (measurement residual)
        Eigen::Vector2d z;
        z << ax_measured, ay_measured;
        Eigen::Vector2d z_pred;
        z_pred << -state_(4), -state_(5);  // Predicted measurement
        Eigen::Vector2d y = z - z_pred;
        
        // Innovation covariance
        Eigen::MatrixXd S = H * P_ * H.transpose() + R_imu_;
        
        // Kalman gain
        Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        
        // Update state
        state_ = state_ + K * y;
        
        // Update covariance
        P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P_;
    }
    
    /**
     * Update step with speed measurement from encoder
     * Uses speed to correct velocity estimates
     */
    void updateWithSpeed(double speed, double yaw) {
        if (!initialized_) return;
        
        // Current velocity magnitude from state
        double v_mag_est = std::sqrt(state_(2)*state_(2) + state_(3)*state_(3));
        
        if (v_mag_est > 0.01) {  // Only update if we have significant velocity
            // Measurement matrix for speed (1x6)
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 6);
            H(0,2) = state_(2) / v_mag_est;  // dvmag/dvx
            H(0,3) = state_(3) / v_mag_est;  // dvmag/dvy
            
            // Innovation
            Eigen::VectorXd y(1);
            y(0) = speed - v_mag_est;
            
            // Innovation covariance
            Eigen::MatrixXd S = H * P_ * H.transpose() + R_speed_;
            
            // Kalman gain
            Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
            
            // Update state
            state_ = state_ + K * y;
            
            // Update covariance
            P_ = (Eigen::MatrixXd::Identity(6, 6) - K * H) * P_;
        } else {
            // If estimated velocity is near zero, directly set velocity based on speed and heading
            state_(2) = speed * std::cos(yaw);
            state_(3) = speed * std::sin(yaw);
        }
    }
    
    Eigen::Vector2d getPosition() const {
        return Eigen::Vector2d(state_(0), state_(1));
    }
    
    Eigen::Vector2d getVelocity() const {
        return Eigen::Vector2d(state_(2), state_(3));
    }
    
    Eigen::Vector2d getAccelerationBias() const {
        return Eigen::Vector2d(state_(4), state_(5));
    }
    
private:
    Eigen::VectorXd state_;      // State vector
    Eigen::MatrixXd P_;          // State covariance
    Eigen::MatrixXd Q_;          // Process noise
    Eigen::MatrixXd R_imu_;      // IMU measurement noise
    Eigen::MatrixXd R_speed_;    // Speed measurement noise
    double dt_;
    bool initialized_;
};

class AdaptivePurePursuit : public rclcpp::Node
{
public:
    AdaptivePurePursuit()
    : Node("adaptive_pure_pursuit_autodrive"),
      heading_angle_(0.5), previous_deviation_(0.0), total_area_(0.0),
      initialized_(false), lookahead_distance_(1.0), a_(1.0), r_(0.8), control_velocity_(0.1),
      gravity_magnitude_(9.81)  // Gravity constant
    {
        // Parameters (tunable)
        max_speed_ = this->declare_parameter("max_speed", 20.0);
        min_speed_ = this->declare_parameter("min_speed", 16.0);
        max_lookahead_ = this->declare_parameter("max_lookahead", 1.3);
        min_lookahead_ = this->declare_parameter("min_lookahead", 1.0);
        wheelbase_ = this->declare_parameter("wheelbase", 0.33);
        beta_ = this->declare_parameter("beta", 0.5);
        heading_scale_ = this->declare_parameter("heading_scale", 1.1);
        area_threshold_ = this->declare_parameter("area_threshold", 1.0);
        speed_factor_ = this->declare_parameter("speed_factor", 0.3);
        velocity_superintendence_1_ = this->declare_parameter("velocity_superintendence_1", 2.1);
        velocity_superintendence_2_ = this->declare_parameter("velocity_superintendence_2", 0.8);
        window_size_ = this->declare_parameter("window_size", 5);
        vel_window = this->declare_parameter("vel_window", 5);
        
        // **Throttle mapping parameters**
        min_throttle_ = this->declare_parameter("min_throttle", 0.15);  // Minimum throttle to overcome friction
        throttle_mapping_type_ = this->declare_parameter("throttle_mapping_type", 2);  // 1=simple, 2=range, 3=nonlinear
        throttle_nonlinear_exponent_ = this->declare_parameter("throttle_nonlinear_exponent", 0.7);

        // Initialize position and IMU data
        current_quaternion_ = {0.0, 0.0, 0.0, 1.0};
        current_speed_ = 0.1;
        
        // **Initialize with given starting position**
        initial_position_ = Eigen::Vector2d(0.7502, 3.1583);
        kalman_filter_.initialize(initial_position_.x(), initial_position_.y());
        current_position_ = initial_position_;
        
        // Initialize gravity vector (will be updated based on IMU orientation)
        gravity_vector_ = Eigen::Vector3d(0, 0, -gravity_magnitude_);

        // Subscribe to sensors (NO IPS subscription anymore)
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/autodrive/f1tenth_1/imu", 10,
            std::bind(&AdaptivePurePursuit::imu_callback, this, std::placeholders::_1));
        speed_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/autodrive/f1tenth_1/speed", 10,
            std::bind(&AdaptivePurePursuit::speed_callback, this, std::placeholders::_1));

        // Publish actuators
        steering_pub_ = this->create_publisher<std_msgs::msg::Float32>("/autodrive/f1tenth_1/steering_command", 10);
        throttle_pub_ = this->create_publisher<std_msgs::msg::Float32>("/autodrive/f1tenth_1/throttle_command", 10);

        // For RViz visualization
        goal_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/goal", 10);
        cp_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/cp", 10);
        race_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/raceline", 10);
        
        // **Add publisher for estimated position visualization**
        est_pos_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/estimated_position", 10);

        // Load the path
        load_raceline_csv("/home/aryan/workspaces/new_ws/src/autodrive_test/new_map_2_modified.csv");

        // Main loop, 100 Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&AdaptivePurePursuit::main_control_loop, this));

        RCLCPP_INFO(this->get_logger(), "AdaptivePurePursuit node with IMU-based position estimation ready.");
        RCLCPP_INFO(this->get_logger(), "Initial position: x=%.4f, y=%.4f", initial_position_.x(), initial_position_.y());
    }

private:
    // ROS handles
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr steering_pub_, throttle_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pub_, cp_pub_, est_pos_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr race_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // State & Params
    std::vector<Eigen::Vector2d> path_;
    std::vector<double> velocities_;
    std::optional<Eigen::Vector2d> previous_position_;
    std::optional<Eigen::Vector2d> current_position_;
    std::array<double, 4> current_quaternion_;
    std::deque<double> area_window_;
    bool initialized_;
    double max_speed_, min_speed_, max_lookahead_, min_lookahead_, wheelbase_;
    double lookahead_distance_, beta_, previous_deviation_, total_area_, control_velocity_, heading_angle_;
    double heading_scale_, area_threshold_, speed_factor_, velocity_superintendence_1_, velocity_superintendence_2_;
    double r_, a_;
    size_t window_size_, vel_window;
    double current_speed_;
    
    // **Throttle mapping parameters**
    double min_throttle_;
    int throttle_mapping_type_;
    double throttle_nonlinear_exponent_;
    
    // **New members for IMU-based position estimation**
    IMUKalmanFilter kalman_filter_;
    Eigen::Vector2d initial_position_;
    Eigen::Vector3d gravity_vector_;
    double gravity_magnitude_;
    rclcpp::Time last_imu_time_;
    bool first_imu_msg_ = true;

    /**
     * IMU callback: Process IMU data for position estimation
     * Steps:
     * 1. Convert quaternion to rotation matrix
     * 2. Transform acceleration from body frame to world frame
     * 3. Compensate for gravity
     * 4. Update Kalman filter with corrected accelerations
     */
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        // Store orientation
        current_quaternion_ = {msg->orientation.x, msg->orientation.y, 
                               msg->orientation.z, msg->orientation.w};
        
        // Get current time
        rclcpp::Time current_time = rclcpp::Time(msg->header.stamp);
        
        if (first_imu_msg_) {
            last_imu_time_ = current_time;
            first_imu_msg_ = false;
            
            // Estimate initial gravity direction from first IMU reading
            Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x, 
                                msg->orientation.y, msg->orientation.z);
            Eigen::Matrix3d R = q.toRotationMatrix();
            
            // Assuming the vehicle starts on level ground, gravity in body frame
            // should be approximately [0, 0, -9.81]
            Eigen::Vector3d gravity_body(0, 0, -gravity_magnitude_);
            gravity_vector_ = R * gravity_body;
            
            return;
        }
        
        // Calculate dt
        double dt = (current_time - last_imu_time_).seconds();
        if (dt <= 0 || dt > 1.0) {  // Sanity check
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                "Invalid dt: %.4f, skipping IMU update", dt);
            return;
        }
        last_imu_time_ = current_time;
        
        // **Transform acceleration from body frame to world frame**
        Eigen::Quaterniond q(msg->orientation.w, msg->orientation.x, 
                            msg->orientation.y, msg->orientation.z);
        Eigen::Matrix3d R = q.toRotationMatrix();
        
        // Get acceleration in body frame
        Eigen::Vector3d accel_body(msg->linear_acceleration.x, 
                                   msg->linear_acceleration.y, 
                                   msg->linear_acceleration.z);
        
        // Transform to world frame
        Eigen::Vector3d accel_world = R * accel_body;
        
        // **Gravity compensation**
        // The IMU measures acceleration including gravity
        // We need to remove gravity to get actual motion acceleration
        // Assuming Z is up, gravity acts downward (-Z direction)
        accel_world.z() += gravity_magnitude_;  // Remove gravity component
        
        // Get yaw angle for heading
        double yaw = quaternion_to_yaw(msg->orientation.x, msg->orientation.y, 
                                       msg->orientation.z, msg->orientation.w);
        
        // **Kalman Filter Prediction and Update**
        // Predict next state using motion model
        kalman_filter_.predict(accel_world.x(), accel_world.y());
        
        // Update with IMU measurements
        kalman_filter_.updateWithIMU(accel_world.x(), accel_world.y());
        
        // Update with speed measurement if available
        if (current_speed_ > 0.01) {  // Only use speed if moving
            kalman_filter_.updateWithSpeed(current_speed_, yaw);
        }
        
        // **Update current position from Kalman filter**
        current_position_ = kalman_filter_.getPosition();
        
        if (!initialized_) {
            previous_position_ = current_position_.value();
            initialized_ = true;
        }
        
        // Debug output (throttled)
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Estimated Position - X: %.3f, Y: %.3f | Velocity - X: %.3f, Y: %.3f | Bias - X: %.3f, Y: %.3f",
            current_position_->x(), current_position_->y(),
            kalman_filter_.getVelocity().x(), kalman_filter_.getVelocity().y(),
            kalman_filter_.getAccelerationBias().x(), kalman_filter_.getAccelerationBias().y());
    }
    
    void speed_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        current_speed_ = msg->data;
    }
    
    void main_control_loop() {
        if (!current_position_.has_value() || !initialized_) return;
        
        // Visualize estimated position
        publish_estimated_position();
        
        double yaw = quaternion_to_yaw(current_quaternion_[0], current_quaternion_[1], 
                                       current_quaternion_[2], current_quaternion_[3]);
        update_lookahead_distance(current_speed_);
        auto [closest_point, goal_point] = find_lookahead_point();
        
        if (goal_point.has_value()) {
            double alpha = calculate_alpha(goal_point.value(), yaw);
            heading_angle_ = calculate_heading_angle(alpha);
            double area = calculate_deviation(current_position_.value(), closest_point);
            double max_velocity_pp = calculate_max_velocity_pure_pursuit(calculate_curvature(alpha));
            double min_deviation_pp = calculate_min_deviation_pure_pursuit(area);
            control_velocity_ = convex_combination(max_velocity_pp, min_deviation_pp, current_speed_, area);
            publish_markers(closest_point, goal_point.value());
            publish_raceline_visualization();
            publish_control_commands();
        }
    }
    
    /**
     * Publish visualization marker for estimated position
     * Shows where the Kalman filter thinks the vehicle is
     */
    void publish_estimated_position() {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.type = visualization_msgs::msg::Marker::ARROW;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Position
        marker.pose.position.x = current_position_->x();
        marker.pose.position.y = current_position_->y();
        marker.pose.position.z = 0.1;
        
        // Orientation (from IMU)
        marker.pose.orientation.x = current_quaternion_[0];
        marker.pose.orientation.y = current_quaternion_[1];
        marker.pose.orientation.z = current_quaternion_[2];
        marker.pose.orientation.w = current_quaternion_[3];
        
        // Scale
        marker.scale.x = 0.3;  // Arrow length
        marker.scale.y = 0.05;  // Arrow width
        marker.scale.z = 0.05;  // Arrow height
        
        // Color - Purple for estimated position
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;
        
        est_pos_pub_->publish(marker);
    }

    void load_raceline_csv(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) { 
            RCLCPP_ERROR(this->get_logger(), "Can't open raceline CSV."); 
            return; 
        }
        std::string line; 
        std::vector<Eigen::Vector2d> temp_path;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string x_str, y_str;
            std::getline(ss, x_str, ','); 
            std::getline(ss, y_str);
            if (x_str.empty() || y_str.empty()) continue;
            double x = std::stod(x_str), y = std::stod(y_str);
            temp_path.emplace_back(x, y);
        }
        if (a_ == 1.0) std::reverse(temp_path.begin(), temp_path.end());
        path_ = temp_path;
    }

    double quaternion_to_yaw(double x, double y, double z, double w) {
        double siny_cosp = 2.0 * (w * z + x * y);
        double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
        return std::atan2(siny_cosp, cosy_cosp);
    }
    
    void update_lookahead_distance(double speed) {
        double normalized_speed = (speed - min_speed_) / (max_speed_ - min_speed_);
        double sigmoid_value = 1.0 / (1.0 + std::exp(-(normalized_speed * 10 - 5)));
        if (speed < min_speed_) lookahead_distance_ = min_lookahead_;
        else lookahead_distance_ = std::min(max_lookahead_, min_lookahead_ + sigmoid_value * (max_lookahead_ - min_lookahead_));
    }
    
    std::pair<Eigen::Vector2d, std::optional<Eigen::Vector2d>> find_lookahead_point() {
        Eigen::Vector2d closest_point; 
        std::optional<Eigen::Vector2d> goal_point;
        double min_dist = std::numeric_limits<double>::max(); 
        size_t closest_idx = 0;
        
        for (size_t i=0; i < path_.size(); ++i) {
            double dist = (path_[i] - current_position_.value()).norm();
            if (dist < min_dist) { 
                min_dist = dist; 
                closest_point = path_[i]; 
                closest_idx = i; 
            }
        }
        
        for (size_t i=closest_idx+2; i < std::min(path_.size(), closest_idx+10); ++i){
            double dist = (path_[i]-current_position_.value()).norm();
            if (dist > lookahead_distance_) { 
                goal_point = path_[i]; 
                break; 
            }
        }
        return {closest_point, goal_point};
    }
    
    double calculate_alpha(const Eigen::Vector2d &goal_point, double yaw) {
        Eigen::Vector2d delta = goal_point - current_position_.value();
        double lx = delta.x() * std::cos(-yaw) - delta.y() * std::sin(-yaw);
        double ly = delta.x() * std::sin(-yaw) + delta.y() * std::cos(-yaw);
        return std::atan2(ly, lx);
    }
    
    double calculate_heading_angle(double alpha) { 
        return std::atan2(2.0 * wheelbase_ * std::sin(alpha), lookahead_distance_); 
    }
    
    double calculate_curvature(double alpha) { 
        return 2.0 * std::sin(alpha) / lookahead_distance_; 
    }
    
    double calculate_deviation(const Eigen::Vector2d &pos, const Eigen::Vector2d &closest) {
        double deviation = (closest - pos).norm();
        if (previous_position_.has_value()) {
            double dist_travel = (pos - previous_position_.value()).norm();
            double area_inc = (deviation + previous_deviation_) / 2.0 * dist_travel;
            area_window_.push_back(area_inc); 
            if (area_window_.size() > window_size_) area_window_.pop_front();
            total_area_ = std::accumulate(area_window_.begin(), area_window_.end(), 0.0);
        }
        previous_position_ = pos; 
        previous_deviation_ = deviation;
        return total_area_;
    }
    
    double calculate_max_velocity_pure_pursuit(double curvature) {
        double max_vel = (curvature != 0.0) ? std::sqrt(1.0 / std::abs(curvature)) : max_speed_;
        return std::min(max_speed_, max_vel);
    }
    
    double calculate_min_deviation_pure_pursuit(double area) { 
        return (area > 0.0) ? max_speed_ / (1.0 + area) : max_speed_; 
    }
    
    double adjust_beta(double current_speed, double area) {
        if (area < area_threshold_) return std::min(1.0, beta_ + 0.25);
        else if (current_speed < max_speed_ * speed_factor_) return std::max(0.0, beta_ - 0.25);
        return beta_;
    }
    
    double convex_combination(double max_v_pp, double min_d_pp, double cur_spd, double area) {
        beta_ = adjust_beta(cur_spd, area);
        double control_v = beta_ * max_v_pp + (1.0 - beta_) * min_d_pp;
        velocities_.push_back(control_v); 
        if (velocities_.size() > vel_window) velocities_.erase(velocities_.begin());
        
        std::vector<double> weights; 
        for (size_t i = 0; i < velocities_.size(); ++i) weights.push_back(std::pow(r_, i));
        double sum_w = std::accumulate(weights.begin(), weights.end(), 0.0); 
        for (auto &w : weights) w /= sum_w;
        
        double moving_avg = 0.0; 
        auto weight_it = weights.rbegin();
        for (auto vel_it = velocities_.rbegin(); vel_it != velocities_.rend(); ++vel_it, ++weight_it)
            moving_avg += (*vel_it) * (*weight_it);
        return moving_avg;
    }
    
    void publish_markers(const Eigen::Vector2d &closest_point, const Eigen::Vector2d &goal_point) {
        auto create_marker = [&](const Eigen::Vector2d &point, float r, float g, float b) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map"; 
            marker.header.stamp = this->get_clock()->now();
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = point.x(); 
            marker.pose.position.y = point.y(); 
            marker.pose.position.z = 0.0;
            marker.scale.x = 0.1; 
            marker.scale.y = 0.1; 
            marker.scale.z = 0.1;
            marker.color.r = r; 
            marker.color.g = g; 
            marker.color.b = b; 
            marker.color.a = 1.0;
            return marker;
        };
        cp_pub_->publish(create_marker(closest_point, 0.0, 0.0, 1.0));
        goal_pub_->publish(create_marker(goal_point, 1.0, 0.0, 0.0));
    }
    
    void publish_raceline_visualization() {
        visualization_msgs::msg::MarkerArray raceline_markers; 
        int id = 0;
        for (const auto &point : path_) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map"; 
            marker.header.stamp = this->get_clock()->now();
            marker.type = visualization_msgs::msg::Marker::SPHERE; 
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = point.x(); 
            marker.pose.position.y = point.y(); 
            marker.pose.position.z = 0.0;
            marker.scale.x = 0.09; 
            marker.scale.y = 0.09; 
            marker.scale.z = 0.09;
            marker.color.r = 0.0; 
            marker.color.g = 1.0; 
            marker.color.b = 0.0; 
            marker.color.a = 1.0;
            marker.id = id++; 
            raceline_markers.markers.push_back(marker);
        }
        race_pub_->publish(raceline_markers);
    }
    
    void publish_control_commands() {
        std_msgs::msg::Float32 steer_cmd;
        steer_cmd.data = std::clamp(heading_angle_ * heading_scale_, -0.24, 0.24);  
        steering_pub_->publish(steer_cmd);
        
        // **Normalize throttle command**
        // control_velocity_ is in m/s (ranging from min_speed_ to max_speed_)
        // We need to convert this to normalized throttle [0, 1]
        
        double normalized_throttle = 0.0;
        
        if (control_velocity_ <= 0) {
            normalized_throttle = 0.0;  // No negative throttle (no reverse)
        } else {
            // Select throttle mapping based on parameter
            switch (throttle_mapping_type_) {
                case 1:  // Simple linear mapping from 0 to max_speed
                    normalized_throttle = control_velocity_ / max_speed_;
                    break;
                    
                case 2:  // Linear mapping from min_speed to max_speed (default)
                    normalized_throttle = (control_velocity_ - min_speed_) / (max_speed_ - min_speed_);
                    break;
                    
                case 3:  // Non-linear mapping for better control at low speeds
                    normalized_throttle = std::pow(control_velocity_ / max_speed_, throttle_nonlinear_exponent_);
                    break;
                    
                default:  // Fallback to type 2
                    normalized_throttle = (control_velocity_ - min_speed_) / (max_speed_ - min_speed_);
                    break;
            }
            
            // Clamp to [0, 1] range
            normalized_throttle = std::max(0.0, std::min(1.0, normalized_throttle));
        }
        
        // Apply minimum throttle to keep the vehicle moving
        // Only apply if we're trying to move (normalized_throttle > 0)
        if (normalized_throttle > 0.01 && normalized_throttle < min_throttle_) {
            normalized_throttle = min_throttle_;
        }
        
        std_msgs::msg::Float32 throttle_cmd;
        throttle_cmd.data = static_cast<float>(normalized_throttle);
        
        // Debug output (throttled to avoid spam)
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "Control: Speed=%.2f m/s, Throttle=%.3f (normalized), Steering=%.3f rad",
            control_velocity_, normalized_throttle, heading_angle_ * heading_scale_);
        
        throttle_pub_->publish(throttle_cmd);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AdaptivePurePursuit>());
    rclcpp::shutdown();
    return 0;
}