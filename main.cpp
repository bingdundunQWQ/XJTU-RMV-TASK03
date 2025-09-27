#include <ceres/ceres.h>
#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

/**
 * @brief 轨迹残差计算结构体
 * 用于计算弹道模型预测值与实际观测值之间的残差
 */
struct TrajectoryResidual {
    TrajectoryResidual(double t, double x_obs, double y_obs, double x0, double y0) 
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}
    
    /**
     * @brief 残差计算函数
     * @param params 参数数组 [vx0, vy0, g, k]
     * @param residual 残差输出数组
     */
    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        const T& vx0 = params[0];  // x方向初速度
        const T& vy0 = params[1];  // y方向初速度
        const T& g = params[2];    // 重力加速度
        const T& k = params[3];    // 阻力系数
        
        // 弹道模型公式：考虑空气阻力的抛体运动
        T x_pred = T(x0_) + (vx0 / k) * (T(1.0) - ceres::exp(-k * T(t_)));
        T y_pred = T(y0_) + ((vy0 + g/k) / k) * (T(1.0) - ceres::exp(-k * T(t_))) - (g/k) * T(t_);
        
        // 计算残差：预测值 - 观测值
        residual[0] = x_pred - T(x_obs_);
        residual[1] = y_pred - T(y_obs_);
        
        return true;
    }
    
private:
    double t_;       // 时间点
    double x_obs_;   // 观测x坐标
    double y_obs_;   // 观测y坐标
    double x0_;      // 初始x位置
    double y0_;      // 初始y位置
};

/**
 * @brief 轨迹拟合器类
 * 负责从视频中提取弹丸轨迹并进行参数拟合
 */
class TrajectoryFitter {
public:
    TrajectoryFitter() : fps_(60.0), frame_count_(0) {}
    
    /**
     * @brief 基于轨迹数据估计初始参数
     * @param params 参数输出数组
     */
    void estimateInitialParameters(double* params) {
        if (time_points_.size() < 10) {
            // 默认初始值
            params[0] = 400.0;  // vx0
            params[1] = 600.0;  // vy0
            params[2] = 500.0;  // g
            params[3] = 0.1;    // k
            return;
        }
        
        // 基于轨迹数据估算速度
        double dx = x_points_.back() - x_points_[0];
        double dt = time_points_.back() - time_points_[0];
        double estimated_vx = std::abs(dx / dt);  // 水平速度估算
        
        // 使用前几帧估算垂直速度（避免重力影响）
        int early_frames = std::min(10, (int)time_points_.size());
        double early_dy = y_points_[early_frames-1] - y_points_[0]; 
        double early_dt = time_points_[early_frames-1] - time_points_[0];
        double estimated_vy = std::abs(early_dy / early_dt);  // 垂直速度估算
        
        // 设置合理的初始参数（考虑阻力影响进行补偿）
        params[0] = estimated_vx * 1.5;  // 水平初速度
        params[1] = estimated_vy * 2.0;  // 垂直初速度
        params[2] = 500.0;               // 重力加速度（中间值）
        params[3] = 0.08;                // 阻力系数
        
        std::cout << "Estimated initial parameters: vx0=" << params[0] 
                  << ", vy0=" << params[1] << std::endl;
    }
    
    /**
     * @brief 从视频文件中提取轨迹点
     * @param video_path 视频文件路径
     * @return 是否成功提取
     */
    bool extractTrajectoryFromVideo(const std::string& video_path) {
        std::cout << "Opening video file: " << video_path << std::endl;
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cout << "ERROR: Cannot open video file: " << video_path << std::endl;
            return false;
        }
        
        // 固定帧率为60FPS
        fps_ = 60.0;
        std::cout << "Video FPS set to: " << fps_ << std::endl;
        
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        std::cout << "Total frames: " << total_frames << std::endl;
        std::cout << "Will process only first 150 frames" << std::endl;
        
        cv::Mat frame;
        frame_count_ = 0;
        
        std::cout << "Processing video frames..." << std::endl;
        
        // 只处理前150帧
        while (cap.read(frame) && frame_count_ < 150) {
            cv::Point2d point = detectProjectile(frame);
            if (point.x >= 0 && point.y >= 0) {
                double t = frame_count_ / fps_;
                time_points_.push_back(t);
                x_points_.push_back(point.x);
                y_points_.push_back(point.y);
                std::cout << "Frame " << frame_count_ << ": detected at (" 
                          << point.x << ", " << point.y << ")" << std::endl;
            } else {
                std::cout << "Frame " << frame_count_ << ": no detection" << std::endl;
            }
            frame_count_++;
            
            // 每10帧输出一次进度
            if (frame_count_ % 10 == 0) {
                std::cout << "Processed " << frame_count_ << " frames..." << std::endl;
            }
        }
        
        cap.release();
        
        // 检查是否提取到足够的轨迹点
        if (time_points_.size() < 10) {
            std::cout << "ERROR: Too few trajectory points extracted: " << time_points_.size() << std::endl;
            return false;
        }
        
        // 数据后处理
        if (time_points_.size() >= 10) {
            removeOutliers();    // 去除异常点
            smoothTrajectory();  // 轨迹平滑
        }
        
        // 设置初始位置
        x0_ = x_points_[0];
        y0_ = y_points_[0];
        
        std::cout << "SUCCESS: Extracted " << time_points_.size() 
                  << " trajectory points from first " << frame_count_ << " frames" << std::endl;
        std::cout << "Initial position: (" << x0_ << ", " << y0_ << ")" << std::endl;
        return true;
    }
    
    /**
     * @brief 执行轨迹参数拟合
     * @param initial_params 初始参数数组（会被优化结果覆盖）
     * @return 拟合是否成功
     */
    bool fitTrajectory(double* initial_params) {
        if (time_points_.empty()) {
            std::cout << "ERROR: No trajectory data" << std::endl;
            return false;
        }
        
        // 自动估计初始参数
        estimateInitialParameters(initial_params);
        
        std::cout << "Starting optimization with " << time_points_.size() << " data points" << std::endl;
        std::cout << "Initial parameters: vx0=" << initial_params[0] << ", vy0=" << initial_params[1] 
                  << ", g=" << initial_params[2] << ", k=" << initial_params[3] << std::endl;
        
        // 创建Ceres优化问题
        ceres::Problem problem;
        
        // 添加所有轨迹点的残差块
        for (size_t i = 0; i < time_points_.size(); ++i) {
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<TrajectoryResidual, 2, 4>(
                    new TrajectoryResidual(time_points_[i], x_points_[i], y_points_[i], x0_, y0_));
            
            problem.AddResidualBlock(cost_function, nullptr, initial_params);
        }
        
        // 设置参数约束条件
        problem.SetParameterLowerBound(initial_params, 2, 100.0);   // g最小值
        problem.SetParameterUpperBound(initial_params, 2, 1000.0);  // g最大值
        problem.SetParameterLowerBound(initial_params, 3, 0.01);    // k最小值
        problem.SetParameterUpperBound(initial_params, 3, 1.0);     // k最大值
        
        // 配置求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;  // 使用稠密QR分解
        options.minimizer_progress_to_stdout = true;   // 输出优化进度
        options.max_num_iterations = 100;              // 最大迭代次数
        options.function_tolerance = 1e-6;             // 函数容忍度
        
        // 执行优化
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        std::cout << summary.FullReport() << std::endl;
        
        return summary.IsSolutionUsable();
    }
    
    /**
     * @brief 检测视频帧中的弹丸位置
     * @param frame 输入帧
     * @return 弹丸中心坐标，未检测到返回(-1, -1)
     */
    cv::Point2d detectProjectile(const cv::Mat& frame) {
        frame_count_++;
        
        // 辅助函数：在二值图像中寻找最大轮廓的中心点
        auto findLargestContourCenter = [](const cv::Mat& binary_img) -> cv::Point2d {
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            if (!contours.empty()) {
                // 寻找面积最大的轮廓
                auto largest_contour = std::max_element(contours.begin(), contours.end(),
                    [](const auto& a, const auto& b) { 
                        return cv::contourArea(a) < cv::contourArea(b); 
                    });
                
                double area = cv::contourArea(*largest_contour);
                
                // 面积过滤，排除噪声和小物体
                if (area > 1.0 && area < 1000) {
                    cv::Moments m = cv::moments(*largest_contour);
                    if (m.m00 != 0) {
                        return cv::Point2d(m.m10/m.m00, m.m01/m.m00);
                    }
                }
            }
            return cv::Point2d(-1, -1);
        };
        
        // 方法1：蓝色通道检测（对蓝色弹丸效果更好）
        cv::Mat channels[3];
        cv::split(frame, channels);
        cv::Mat blue_thresh;
        cv::threshold(channels[0], blue_thresh, 120, 255, cv::THRESH_BINARY);
        cv::Point2d blue_result = findLargestContourCenter(blue_thresh);
        
        // 方法2：灰度图检测（备用方法）
        cv::Mat gray, gray_thresh;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, gray_thresh, 80, 255, cv::THRESH_BINARY);
        cv::Point2d gray_result = findLargestContourCenter(gray_thresh);
        
        // 优先使用蓝色通道结果，失败时使用灰度结果
        cv::Point2d final_result = blue_result.x >= 0 ? blue_result : gray_result;
        std::string method = blue_result.x >= 0 ? "Blue" : (gray_result.x >= 0 ? "Gray" : "None");
        
        if (final_result.x >= 0) {
            std::cout << "Frame " << frame_count_ << " [" << method << "]: (" 
                      << final_result.x << ", " << final_result.y << ")" << std::endl;
        } else {
            std::cout << "Frame " << frame_count_ << ": no detection" << std::endl;
        }
        
        return final_result;
    }
    
    /**
     * @brief 手动设置轨迹数据（用于测试）
     */
    void setTrajectoryData(const std::vector<double>& t, 
                          const std::vector<double>& x, 
                          const std::vector<double>& y) {
        time_points_ = t;
        x_points_ = x;
        y_points_ = y;
        
        if (!t.empty()) {
            x0_ = x[0];
            y0_ = y[0];
        }
    }
    
    /**
     * @brief 打印轨迹数据摘要
     */
    void printTrajectoryData() {
        std::cout << "Trajectory Data Summary:" << std::endl;
        std::cout << "Time(s)\tX\tY" << std::endl;
        for (size_t i = 0; i < time_points_.size(); ++i) {
            std::cout << time_points_[i] << "\t" << x_points_[i] << "\t" << y_points_[i] << std::endl;
        }
    }
    
private:
    std::vector<double> time_points_;  // 时间点序列
    std::vector<double> x_points_;     // x坐标序列
    std::vector<double> y_points_;     // y坐标序列
    double x0_, y0_;                   // 初始位置
    double fps_;                       // 视频帧率
    int frame_count_;                  // 帧计数器
    
    /**
     * @brief 去除轨迹数据中的异常点
     */
    void removeOutliers() {
        if (time_points_.size() < 3) return;
        
        std::vector<double> speeds;              // 速度序列
        std::vector<bool> is_valid(time_points_.size(), true);
        is_valid[0] = true;  // 第一个点总是有效
        
        // 计算相邻帧间的速度
        for (size_t i = 1; i < time_points_.size(); i++) {
            double dx = x_points_[i] - x_points_[i-1];
            double dy = y_points_[i] - y_points_[i-1];
            double dt = time_points_[i] - time_points_[i-1];
            if (dt > 0.001) {  // 避免除零
                double speed = sqrt(dx*dx + dy*dy) / dt;
                speeds.push_back(speed);
            }
        }
        
        if (speeds.empty()) return;
        
        // 计算速度的中位数和标准差
        std::vector<double> sorted_speeds = speeds;
        std::sort(sorted_speeds.begin(), sorted_speeds.end());
        double median_speed = sorted_speeds[sorted_speeds.size()/2];
        
        double sum_sq = 0;
        for (double speed : speeds) {
            sum_sq += (speed - median_speed) * (speed - median_speed);
        }
        double std_speed = sqrt(sum_sq / speeds.size());
        
        // 标记速度异常的点（超过3倍标准差）
        for (size_t i = 1; i < time_points_.size(); i++) {
            if (i-1 < speeds.size()) {
                double speed = speeds[i-1];
                if (std::abs(speed - median_speed) > 3 * std_speed) {
                    is_valid[i] = false;
                }
            }
        }
        
        // 创建新的轨迹数据，排除异常点
        std::vector<double> new_time, new_x, new_y;
        for (size_t i = 0; i < time_points_.size(); i++) {
            if (is_valid[i]) {
                new_time.push_back(time_points_[i]);
                new_x.push_back(x_points_[i]);
                new_y.push_back(y_points_[i]);
            }
        }
        
        // 如果保留了足够多的点，则更新数据
        if (new_time.size() >= time_points_.size() * 0.8) {
            int removed_count = time_points_.size() - new_time.size();
            time_points_ = new_time;
            x_points_ = new_x;
            y_points_ = new_y;
            std::cout << "Removed " << removed_count << " outliers" << std::endl;
        }
    }
    
    /**
     * @brief 对轨迹数据进行平滑处理
     */
    void smoothTrajectory() {
        if (time_points_.size() < 5) return;
        
        std::vector<double> smoothed_x = x_points_;
        std::vector<double> smoothed_y = y_points_;
        
        // 使用5点移动平均进行平滑
        for (size_t i = 2; i < time_points_.size() - 2; i++) {
            smoothed_x[i] = (x_points_[i-2] + x_points_[i-1] + x_points_[i] + 
                           x_points_[i+1] + x_points_[i+2]) / 5.0;
            smoothed_y[i] = (y_points_[i-2] + y_points_[i-1] + y_points_[i] + 
                           y_points_[i+1] + y_points_[i+2]) / 5.0;
        }
        
        x_points_ = smoothed_x;
        y_points_ = smoothed_y;
        std::cout << "Applied trajectory smoothing" << std::endl;
    }
};

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    // 初始化Google日志系统
    google::InitGoogleLogging(argv[0]);
    
    TrajectoryFitter fitter;
    std::string video_path;
    
    std::cout << "put in your video path: " << std::endl;
    std::getline(std::cin,video_path);

    //std::string video_path = "/media/psf/Home/Desktop/test /resource/video.mp4";
    
    // 显示程序信息
    std::cout << "==========================================" << std::endl;
    std::cout << "Trajectory Fitting from Video" << std::endl;
    std::cout << "Video path: " << video_path << std::endl;
    std::cout << "FPS: 60 (fixed)" << std::endl;
    std::cout << "Processing first 150 frames only" << std::endl;
    std::cout << "Parameter constraints: g[100, 1000], k[0.01, 1.0]" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // 从视频提取轨迹
    if (fitter.extractTrajectoryFromVideo(video_path)) {
        std::cout << "Video processing successful!" << std::endl;
        
        double initial_params[4] = {400.0, 600.0, 500.0, 0.1};
        
        std::cout << "Starting trajectory fitting..." << std::endl;
        
        // 执行参数拟合
        if (fitter.fitTrajectory(initial_params)) {
            std::cout << "Fitting successful!" << std::endl;
            std::cout << "Final parameters: vx0=" << initial_params[0] << " px/s, "
                      << "vy0=" << initial_params[1] << " px/s, "
                      << "g=" << initial_params[2] << " px/s², "
                      << "k=" << initial_params[3] << " 1/s" << std::endl;
            
            // 验证参数是否符合要求
            if (initial_params[2] >= 100 && initial_params[2] <= 1000 && 
                initial_params[3] >= 0.01 && initial_params[3] <= 1.0) {
                std::cout << "All parameters within required constraints!" << std::endl;
            } else {
                std::cout << "WARNING: Some parameters outside required constraints!" << std::endl;
            }
        } else {
            std::cout << "Fitting failed!" << std::endl;
            return -1;
        }
    } else {
        std::cout << "Video processing failed!" << std::endl;
        return -1;
    }
    
    return 0;
}