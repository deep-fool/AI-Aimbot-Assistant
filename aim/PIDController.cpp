//
// Created by xiang on 2026/5/14.
//

#include "PIDController.h"

#include <algorithm>
#include <cmath>

namespace
{
constexpr float kDefaultDeltaTimeMs = 10.0f;
constexpr float kMinDeltaTimeMs = 1.0f;
constexpr float kMaxDeltaTimeMs = 50.0f;
constexpr float kVelocityAlpha = 0.25f;
constexpr float kAccelerationAlpha = 0.15f;
constexpr float kMaxVelocity = 3000.0f;
constexpr float kMaxAcceleration = 5000.0f;
constexpr float kIntegralLimit = 100.0f;
constexpr float kDerivativeLimit = 50.0f;
}

PIDController::PIDController(float kp, float ki, float kd)
    : kp_(kp)
    , ki_(ki)
    , kd_(kd)
    , rawAimPos_(0.0f)
    , predictionWeight_(0.0f)
    , initScale_(1.0f)
    , rampTimeMs_(0.0f)
    , integralTerm_(0.0f)
    , previousFusionError_(0.0f)
    , previousRawError_(0.0f)
    , maxOrderTaylor_(2)
    , smoothedDerivatives_(2, 0.0f)
    , previousSmoothedDerivatives_(2, 0.0f)
    , lastOutput_(0.0f)
    , hasPreviousFusionError_(false)
    , hasPreviousRawError_(false)
    , hasLastOutput_(false)
    , hasLockStartTime_(false)
    , hasLastUpdateTime_(false)
    , minOutput_(-std::numeric_limits<float>::infinity())
    , maxOutput_(std::numeric_limits<float>::infinity())
{
}

void PIDController::setPID(float kp, float ki, float kd)
{
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
}

void PIDController::setKp(float kp)
{
    kp_ = kp;
}

void PIDController::setKi(float ki)
{
    ki_ = ki;
}

void PIDController::setKd(float kd)
{
    kd_ = kd;
}

void PIDController::setAimPos(float aimPos)
{
    rawAimPos_ = aimPos;
}

void PIDController::setPredictionWeight(float predictionWeight)
{
    predictionWeight_ = predictionWeight;
}

void PIDController::setPredictionMaxOrderTaylor(float maxOrderTaylor)
{
    maxOrderTaylor_ = maxOrderTaylor < 2.0f ? 1 : static_cast<int>(std::floor(maxOrderTaylor));
    smoothedDerivatives_.assign(maxOrderTaylor_, 0.0f);
    previousSmoothedDerivatives_.assign(maxOrderTaylor_, 0.0f);
}

void PIDController::setInitScale(float initScale)
{
    initScale_ = clamp(initScale, 0.0f, 1.0f);
}

void PIDController::setRampTimeMs(float rampTimeMs)
{
    rampTimeMs_ = std::max(0.0f, rampTimeMs);
}

void PIDController::setOutputLimits(float minOutput, float maxOutput)
{
    if (minOutput > maxOutput) {
        std::swap(minOutput, maxOutput);
    }

    minOutput_ = minOutput;
    maxOutput_ = maxOutput;
}

void PIDController::reset()
{
    integralTerm_ = 0.0f;
    previousFusionError_ = 0.0f;
    previousRawError_ = 0.0f;
    std::fill(smoothedDerivatives_.begin(), smoothedDerivatives_.end(), 0.0f);
    std::fill(previousSmoothedDerivatives_.begin(), previousSmoothedDerivatives_.end(), 0.0f);
    lastOutput_ = 0.0f;
    hasPreviousFusionError_ = false;
    hasPreviousRawError_ = false;
    hasLastOutput_ = false;
    hasLockStartTime_ = false;
    hasLastUpdateTime_ = false;
}

float PIDController::update(float measurement, float deltaTimeMs)
{
    const auto now = std::chrono::steady_clock::now();

    float resolvedDeltaTimeMs = deltaTimeMs;
    if (resolvedDeltaTimeMs <= 0.0f) {
        if (!hasLastUpdateTime_) {
            resolvedDeltaTimeMs = kDefaultDeltaTimeMs;
        } else {
            resolvedDeltaTimeMs = std::chrono::duration<float, std::milli>(now - lastUpdateTime_).count();
        }
    }
    resolvedDeltaTimeMs = clamp(resolvedDeltaTimeMs, kMinDeltaTimeMs, kMaxDeltaTimeMs);
    const float deltaTime = resolvedDeltaTimeMs / 1000.0f;

    const float currentRawError = rawAimPos_ - measurement;
    float predictedDisplacement = 0.0f;

    if (hasPreviousRawError_) {
        // 上一帧输出会抵消掉一部分目标真实位移，因此这里要把它加回去估算目标运动趋势。
        float velocityRaw = ((currentRawError - previousRawError_) + (hasLastOutput_ ? lastOutput_ : 0.0f)) / deltaTime;
        velocityRaw = clamp(velocityRaw, -kMaxVelocity, kMaxVelocity);

        if (std::fabs(currentRawError) > 5.0f && velocityRaw * currentRawError < 0.0f) {
            velocityRaw *= 0.1f;
        }

        if (static_cast<int>(smoothedDerivatives_.size()) != maxOrderTaylor_) {
            smoothedDerivatives_.assign(maxOrderTaylor_, 0.0f);
            previousSmoothedDerivatives_.assign(maxOrderTaylor_, 0.0f);
        }

        previousSmoothedDerivatives_ = smoothedDerivatives_;
        const float velocityAlpha = adaptiveAlpha(kVelocityAlpha, deltaTime);
        smoothedDerivatives_[0] = velocityAlpha * velocityRaw + (1.0f - velocityAlpha) * smoothedDerivatives_[0];

        const float accelerationAlpha = adaptiveAlpha(kAccelerationAlpha, deltaTime);
        for (int order = 2; order <= maxOrderTaylor_; ++order) {
            float derivativeRaw = (smoothedDerivatives_[order - 2] - previousSmoothedDerivatives_[order - 2]) / deltaTime;
            if (order == 2) {
                derivativeRaw = clamp(derivativeRaw, -kMaxAcceleration, kMaxAcceleration);
                if (std::fabs(currentRawError) > 5.0f && derivativeRaw * currentRawError < 0.0f) {
                    derivativeRaw *= 0.1f;
                }
            }
            smoothedDerivatives_[order - 1] =
                accelerationAlpha * derivativeRaw + (1.0f - accelerationAlpha) * smoothedDerivatives_[order - 1];
        }

        float taylorCoefficient = 1.0f;
        for (int order = 1; order <= maxOrderTaylor_; ++order) {
            taylorCoefficient *= deltaTime / static_cast<float>(order);
            predictedDisplacement += smoothedDerivatives_[order - 1] * taylorCoefficient;
        }

        const float maxPrediction = std::min(std::max(std::fabs(currentRawError) * 1.5f, 30.0f), 60.0f);
        predictedDisplacement = clamp(predictedDisplacement, -maxPrediction, maxPrediction);
    }

    const float fusionError = currentRawError + predictedDisplacement * predictionWeight_;

    if (!hasLockStartTime_) {
        lockStartTime_ = now;
        hasLockStartTime_ = true;
    }

    float kpScale = 1.0f;
    if (rampTimeMs_ > 0.0f) {
        const float elapsedMs = std::chrono::duration<float, std::milli>(now - lockStartTime_).count();
        if (elapsedMs < rampTimeMs_) {
            const float progress = clamp(elapsedMs / rampTimeMs_, 0.0f, 1.0f);
            kpScale = initScale_ + (1.0f - initScale_) * progress;
        }
    }

    const float proportionalTerm = fusionError * kp_ * kpScale;
    integralTerm_ += fusionError * deltaTime * ki_;
    integralTerm_ = clamp(integralTerm_, -kIntegralLimit, kIntegralLimit);

    float derivativeTerm = 0.0f;
    if (hasPreviousFusionError_) {
        derivativeTerm = (fusionError - previousFusionError_) / deltaTime * kd_;
        derivativeTerm = clamp(derivativeTerm, -kDerivativeLimit, kDerivativeLimit);
    }

    const float output = clamp(proportionalTerm + integralTerm_ + derivativeTerm, minOutput_, maxOutput_);

    previousFusionError_ = fusionError;
    previousRawError_ = currentRawError;
    lastOutput_ = output;
    hasPreviousFusionError_ = true;
    hasPreviousRawError_ = true;
    hasLastOutput_ = true;
    lastUpdateTime_ = now;
    hasLastUpdateTime_ = true;

    return output;
}

float PIDController::clamp(float value, float minValue, float maxValue) const
{
    return std::max(minValue, std::min(value, maxValue));
}

float PIDController::adaptiveAlpha(float baseAlpha, float deltaTimeSeconds) const
{
    const float alpha = 1.0f - std::pow(1.0f - baseAlpha, deltaTimeSeconds / 0.01f);
    return clamp(alpha, 0.05f, 0.8f);
}


void PIDController::resetIntegral()
{
    integralTerm_ = 0.0f;
}