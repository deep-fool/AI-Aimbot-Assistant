//
// Created by xiang on 2026/5/14.
//

#ifndef PID_PIDCONTROLLER_H
#define PID_PIDCONTROLLER_H

#include <chrono>
#include <limits>
#include <vector>

class PIDController
{
public:
    /**
     * @brief 构造单轴 PID 控制器并设置初始 PID 参数。
     * @param kp 比例系数。
     * @param ki 积分系数。
     * @param kd 微分系数。
     */
    PIDController(float kp, float ki, float kd);

    /**
     * @brief 一次性更新 PID 三个核心参数。
     * @param kp 比例系数。
     * @param ki 积分系数。
     * @param kd 微分系数。
     */
    void setPID(float kp, float ki, float kd);

    /**
     * @brief 单独设置比例系数。
     * @param kp 比例系数。
     */
    void setKp(float kp);

    void resetIntegral();

    /**
     * @brief 单独设置积分系数。
     * @param ki 积分系数。
     */
    void setKi(float ki);

    /**
     * @brief 单独设置微分系数。
     * @param kd 微分系数。
     */
    void setKd(float kd);

    /**
     * @brief 设置当前帧目标相对中心的原始偏移。
     * @param aimPos 当前帧目标相对中心的偏移量。
     */
    void setAimPos(float aimPos);

    /**
     * @brief 设置预测位移参与误差融合的权重。
     * @param predictionWeight 预测位移权重。
     */
    void setPredictionWeight(float predictionWeight);

    /**
     * @brief 设置泰勒预测最高展开阶数，默认 2 阶表示使用速度和加速度，小于 2 时按 1 阶处理。
     * @param maxOrderTaylor 泰勒展开最高阶数。
     */
    void setPredictionMaxOrderTaylor(float maxOrderTaylor = 2.0f);

    /**
     * @brief 设置锁定初期的比例系数缩放值。
     * @param initScale 锁定刚开始时的 P 缩放比例。
     */
    void setInitScale(float initScale);

    /**
     * @brief 设置比例系数从初始缩放爬升到完整值所需的时间。
     * @param rampTimeMs P 值爬升时间，单位为毫秒。
     */
    void setRampTimeMs(float rampTimeMs);

    /**
     * @brief 设置最终输出上下限。
     * @param minOutput 最小输出。
     * @param maxOutput 最大输出。
     */
    void setOutputLimits(float minOutput, float maxOutput);

    /**
     * @brief 重置积分、预测器和平滑启动相关状态。
     */
    void reset();

    /**
     * @brief 计算当前帧控制输出。
     * @param measurement 当前测量值。
     * @param deltaTimeMs 大于 0 时使用固定采样周期，小于等于 0 时自动计算真实帧间隔。
     * @return 当前帧建议输出量。
     */
    float update(float measurement, float deltaTimeMs);

private:
    /**
     * @brief 将数值限制在指定区间内。
     * @param value 待限制的数值。
     * @param minValue 区间下限。
     * @param maxValue 区间上限。
     * @return 限制后的数值。
     */
    float clamp(float value, float minValue, float maxValue) const;

    /**
     * @brief 根据采样周期计算自适应 EMA 系数。
     * @param baseAlpha 以 10ms 为基准的原始 EMA 系数。
     * @param deltaTimeSeconds 当前采样周期，单位为秒。
     * @return 适配当前采样周期后的 EMA 系数。
     */
    float adaptiveAlpha(float baseAlpha, float deltaTimeSeconds) const;

private:
    float kp_;
    float ki_;
    float kd_;
    float rawAimPos_;
    float predictionWeight_;
    float initScale_;
    float rampTimeMs_;
    float integralTerm_;
    float previousFusionError_;
    float previousRawError_;
    int maxOrderTaylor_;
    std::vector<float> smoothedDerivatives_;
    std::vector<float> previousSmoothedDerivatives_;
    float lastOutput_;
    bool hasPreviousFusionError_;
    bool hasPreviousRawError_;
    bool hasLastOutput_;
    bool hasLockStartTime_;
    bool hasLastUpdateTime_;
    float minOutput_;
    float maxOutput_;
    std::chrono::steady_clock::time_point lockStartTime_;
    std::chrono::steady_clock::time_point lastUpdateTime_;
};

#endif // PID_PIDCONTROLLER_H
