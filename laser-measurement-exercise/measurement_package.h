#pragma once

#include "../Eigen/Dense"

class MeasurementPackage {
public:

    enum SensorType {
        LASER, 
        RADAR
    } sensor_type_;

    Eigen::VectorXd raw_measurements_;

    int64_t timestamp_;

};
