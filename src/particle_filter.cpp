/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;
using std::normal_distribution;
using std::default_random_engine;
std::default_random_engine gen;
using vector_t = std::vector<double>;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Add random Gaussian noise to each particle.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  num_particles = 500;
  for(int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0f;
    particles.push_back(p);
    weights.push_back(1.0f);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  // define normal distributions for sensor noise

  // create normal (Gaussian) distribution for x, y and theta
  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    // add normal distribution noise
    normal_distribution<double> N_x(0, std_pos[0]);
    normal_distribution<double> N_y(0, std_pos[1]);
    normal_distribution<double> N_theta(0, std_pos[2]);
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.
  double min_distance, dist, dx, dy;
  int min_i;
  for(unsigned int obs_i = 0; obs_i < observations.size(); ++obs_i) {
    auto obs = observations[obs_i];
    min_distance = 10000000.0f;
    min_i = -1;
    for(unsigned int i = 0; i < predicted.size(); i++) {
      auto pred_lm = predicted[i];
      dx = (pred_lm.x - obs.x);
      dy = (pred_lm.y - obs.y);
      dist = dx * dx + dy * dy;
      if(dist < min_distance) {
        min_distance = dist;
        min_i = i;
      }
    }
    observations[obs_i].id = min_i;
  }
}

const LandmarkObs ParticleFilter::transCoordsFromCarToMap(const LandmarkObs& obs, const Particle& p) {
  // transform car coordinate observation into map coordinate
  LandmarkObs out;
  out.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
  out.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
  out.id = obs.id;
  return out;
}

const double ParticleFilter::multivariateGaussian(const LandmarkObs& obs, const LandmarkObs &lm, const double sigma[]) {
  double cov_x = sigma[0] * sigma[0];
  double cov_y = sigma[1] * sigma[1];
  double gauss_norm = 1/(2.0 * M_PI * sigma[0] * sigma[1]);
  double dx = (obs.x - lm.x);
  double dy = (obs.y - lm.y);
  return exp(-(dx * dx / (2 * cov_x) + dy * dy/(2 * cov_y))) * gauss_norm;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  // Landmark measurement uncertainty [x [m], y [m]]
  double sigma_landmark [2] = {0.3, 0.3}; 
  for(unsigned int p_ctr=0; p_ctr < particles.size(); ++p_ctr) {
    auto p = particles[p_ctr];
    std::vector<LandmarkObs> predicted_landmarks;
    for(auto lm : map_landmarks.landmark_list) {
      LandmarkObs lm_pred;
      lm_pred.x = lm.x_f;
      lm_pred.y = lm.y_f;
      lm_pred.id = lm.id_i;
      auto dx = lm_pred.x - p.x;
      auto dy = lm_pred.y - p.y;
      // If the dist is in the sensor range, add it
      if((dx * dx + dy * dy) <= sensor_range * sensor_range)
        predicted_landmarks.push_back(lm_pred);
    }
    std::vector<LandmarkObs> transformed_obs;
    double total_prob = 1.0f;
    // Transform coordinates of all observations (for current particle)
    for(auto obs_lm : observations) {
      auto obs_global = transCoordsFromCarToMap(obs_lm, p);
      transformed_obs.push_back(std::move(obs_global));
    }
    // Stores index of associated landmark in the observation
    dataAssociation(predicted_landmarks, transformed_obs);
    for(unsigned int i = 0; i < transformed_obs.size(); ++i) {
      auto obs = transformed_obs[i];
      // Assume sorted by id and starting at 1
      auto assoc_lm = predicted_landmarks[obs.id];
      double pdf = multivariateGaussian(obs, assoc_lm, sigma_landmark);
      total_prob *= pdf;
    }
    particles[p_ctr].weight = total_prob;
    weights[p_ctr] = total_prob;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::discrete_distribution<int> d(weights.begin(), weights.end());
  std::vector<Particle> new_particles;

  for(int i = 0; i < num_particles; i++) {
    auto ind = d(gen);
    new_particles.push_back(std::move(particles[ind]));
  }
  particles = std::move(new_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
