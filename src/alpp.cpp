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

class AdaptivePurePursuit : public rclcpp::Node
{
public:
    AdaptivePurePursuit()
    : Node("adaptive_pure_pursuit_autodrive"),
      heading_angle_(0.5), previous_deviation_(0.0), total_area_(0.0),
      initialized_(false), lookahead_distance_(1.0), a_(1.0), r_(0.8), control_velocity_(0.1)
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

        current_quaternion_ = {0.0, 0.0, 0.0, 1.0};
        current_speed_ = 0.1;

        // Subscribe to sensors
        pos_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/autodrive/f1tenth_1/ips", 10,
            std::bind(&AdaptivePurePursuit::ips_callback, this, std::placeholders::_1));
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

        // Load the path
        load_raceline_csv("/home/aryan/workspaces/new_ws/src/autodrive_test/new_map_2_modified.csv");

        // Main loop, 100 Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&AdaptivePurePursuit::main_control_loop, this));

        RCLCPP_INFO(this->get_logger(), "AdaptivePurePursuit node (autodrive sim) ready.");
    }

private:
    // ROS handles
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr pos_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr steering_pub_, throttle_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pub_, cp_pub_;
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

    void ips_callback(const geometry_msgs::msg::Point::SharedPtr msg) {
        current_position_ = Eigen::Vector2d(msg->x, msg->y);
        if (!initialized_) { previous_position_ = current_position_.value(); initialized_ = true; }
    }
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        current_quaternion_ = {msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w};
    }
    void speed_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        current_speed_ = msg->data;
    }
    void main_control_loop() {
        if (!current_position_.has_value() || !initialized_) return;
        double yaw = quaternion_to_yaw(current_quaternion_[0], current_quaternion_[1], current_quaternion_[2], current_quaternion_[3]);
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

    void load_raceline_csv(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) { RCLCPP_ERROR(this->get_logger(), "Can't open raceline CSV."); return; }
        std::string line; std::vector<Eigen::Vector2d> temp_path;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string x_str, y_str;
            std::getline(ss, x_str, ','); std::getline(ss, y_str);
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
        Eigen::Vector2d closest_point; std::optional<Eigen::Vector2d> goal_point;
        double min_dist = std::numeric_limits<double>::max(); size_t closest_idx = 0;
        for (size_t i=0; i < path_.size(); ++i) {
            double dist = (path_[i] - current_position_.value()).norm();
            if (dist < min_dist) { min_dist = dist; closest_point = path_[i]; closest_idx = i; }
        }
        for (size_t i=closest_idx+2; i < std::min(path_.size(), closest_idx+10); ++i){
            double dist = (path_[i]-current_position_.value()).norm();
            if (dist > lookahead_distance_) { goal_point = path_[i]; break; }
        }
        return {closest_point, goal_point};
    }
    double calculate_alpha(const Eigen::Vector2d &goal_point, double yaw) {
        Eigen::Vector2d delta = goal_point - current_position_.value();
        double lx = delta.x() * std::cos(-yaw) - delta.y() * std::sin(-yaw);
        double ly = delta.x() * std::sin(-yaw) + delta.y() * std::cos(-yaw);
        return std::atan2(ly, lx);
    }
    double calculate_heading_angle(double alpha) { return std::atan2(2.0 * wheelbase_ * std::sin(alpha), lookahead_distance_); }
    double calculate_curvature(double alpha) { return 2.0 * std::sin(alpha) / lookahead_distance_; }
    double calculate_deviation(const Eigen::Vector2d &pos, const Eigen::Vector2d &closest) {
        double deviation = (closest - pos).norm();
        if (previous_position_.has_value()) {
            double dist_travel = (pos - previous_position_.value()).norm();
            double area_inc = (deviation + previous_deviation_) / 2.0 * dist_travel;
            area_window_.push_back(area_inc); if (area_window_.size() > window_size_) area_window_.pop_front();
            total_area_ = std::accumulate(area_window_.begin(), area_window_.end(), 0.0);
        }
        previous_position_ = pos; previous_deviation_ = deviation;
        return total_area_;
    }
    double calculate_max_velocity_pure_pursuit(double curvature) {
        double max_vel = (curvature != 0.0) ? std::sqrt(1.0 / std::abs(curvature)) : max_speed_;
        return std::min(max_speed_, max_vel);
    }
    double calculate_min_deviation_pure_pursuit(double area) { return (area > 0.0) ? max_speed_ / (1.0 + area) : max_speed_; }
    double adjust_beta(double current_speed, double area) {
        if (area < area_threshold_) return std::min(1.0, beta_ + 0.25);
        else if (current_speed < max_speed_ * speed_factor_) return std::max(0.0, beta_ - 0.25);
        return beta_;
    }
    double convex_combination(double max_v_pp, double min_d_pp, double cur_spd, double area) {
        beta_ = adjust_beta(cur_spd, area);
        double control_v = beta_ * max_v_pp + (1.0 - beta_) * min_d_pp;
        velocities_.push_back(control_v); if (velocities_.size() > vel_window) velocities_.erase(velocities_.begin());
        std::vector<double> weights; for (size_t i = 0; i < velocities_.size(); ++i) weights.push_back(std::pow(r_, i));
        double sum_w = std::accumulate(weights.begin(), weights.end(), 0.0); for (auto &w : weights) w /= sum_w;
        double moving_avg = 0.0; auto weight_it = weights.rbegin();
        for (auto vel_it = velocities_.rbegin(); vel_it != velocities_.rend(); ++vel_it, ++weight_it)
            moving_avg += (*vel_it) * (*weight_it);
        return moving_avg;
    }
    void publish_markers(const Eigen::Vector2d &closest_point, const Eigen::Vector2d &goal_point) {
        auto create_marker = [&](const Eigen::Vector2d &point, float r, float g, float b) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map"; marker.header.stamp = this->get_clock()->now();
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = point.x(); marker.pose.position.y = point.y(); marker.pose.position.z = 0.0;
            marker.scale.x = 0.1; marker.scale.y = 0.1; marker.scale.z = 0.1;
            marker.color.r = r; marker.color.g = g; marker.color.b = b; marker.color.a = 1.0;
            return marker;
        };
        cp_pub_->publish(create_marker(closest_point, 0.0, 0.0, 1.0));
        goal_pub_->publish(create_marker(goal_point, 1.0, 0.0, 0.0));
    }
    void publish_raceline_visualization() {
        visualization_msgs::msg::MarkerArray raceline_markers; int id = 0;
        for (const auto &point : path_) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "map"; marker.header.stamp = this->get_clock()->now();
            marker.type = visualization_msgs::msg::Marker::SPHERE; marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = point.x(); marker.pose.position.y = point.y(); marker.pose.position.z = 0.0;
            marker.scale.x = 0.09; marker.scale.y = 0.09; marker.scale.z = 0.09;
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0;
            marker.id = id++; raceline_markers.markers.push_back(marker);
        }
        race_pub_->publish(raceline_markers);
    }
    void publish_control_commands() {
        std_msgs::msg::Float32 steer_cmd;
        steer_cmd.data = std::clamp(heading_angle_ * heading_scale_, -0.24, 0.24);  
        steering_pub_->publish(steer_cmd);
        std_msgs::msg::Float32 throttle_cmd;
        throttle_cmd.data = static_cast<float>(control_velocity_);
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
