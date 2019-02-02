#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::numeric_limits;
using std::sqrt;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse = VectorXd::Zero(4);

  // Ensure estimate vector size and ground truth vector size are the same.
  if (estimations.size() != ground_truth.size()
      || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // Calculate square residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd res = estimations[i] - ground_truth[i];
    // coefficient-wise multiplication
    res = res.array() * res.array();
    rmse += res;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {

  MatrixXd Hj(3,4);

  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // confirm px, py aren't zero
  if ((px <= numeric_limits<float>::epsilon()) &&
      (py <= numeric_limits<float>::epsilon())) {
    cout << "px and py can't both be zero!" << endl;
    return Hj;
  }

  float c1 = pow(px, 2) + pow(py, 2);
  float c2 = sqrt(c1);

  // compute the Jacobian matrix
  Hj(0, 0) = px / c2;
  Hj(0, 1) = py / c2;
  Hj(0, 2) = 0;
  Hj(0, 3) = 0;

  Hj(1, 0) = (-1 * py) / c1;
  Hj(1, 1) = px / c1;
  Hj(1, 2) = 0;
  Hj(1, 3) = 0;

  Hj(2, 0) = (py * ((vx * py) - (vy * px))) / pow(c1, 1.5);
  Hj(2, 1) = (px * ((vy * px) - (vx * py))) / pow(c1, 1.5);
  Hj(2, 2) = px / c2;
  Hj(2, 3) = py / c2;

  return Hj;
}

VectorXd Tools::CalcRadarMeasurements(const VectorXd &x_state, const VectorXd &z) {
  VectorXd radar(3);

  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float rho = sqrt((px * px) + (py * py));

  float phi = atan(py / px);
  while (phi - z(1) > M_PI / 2)
    phi -= M_PI;
  while (z(1) - phi > M_PI / 2)
    phi += M_PI;

  float rho_dot = ((px * vx) + (py * vy)) / rho;

  radar << rho, phi, rho_dot;
  return radar;
}
