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
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  particles.resize(num_particles);
  for (int i=0; i<num_particles; i++) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  for (int i=0; i<num_particles; i++) {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    double thetaf, xf, yf;
    
    if (fabs(yaw_rate) < 0.0001) {
      // simple kinematics
      thetaf = theta;
      xf = x + velocity * cos(theta) * delta_t;
      yf = y + velocity * sin(theta) * delta_t;
    }
    else {
      // bicycle model kinematics
      thetaf = theta + yaw_rate*delta_t;
      xf = x + velocity/yaw_rate * (sin(thetaf) - sin(theta));
      yf = y + velocity/yaw_rate * (cos(theta) - cos(thetaf));
    }

    // add Gaussian noise to estimated position
    normal_distribution<double> dist_x(xf, std_pos[0]);
    normal_distribution<double> dist_y(yf, std_pos[1]);
    normal_distribution<double> dist_theta(thetaf, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  int obs_size = observations.size();

  for (int i=0; i<obs_size; i++) {
    double mindist = std::numeric_limits<double>::max();
    int id_match = -1;
    for (int j=0; j<obs_size; j++) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < mindist) {
        mindist = distance;
        id_match = predicted[j].id;
      }
    }
    observations[i].id = id_match;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   vector<LandmarkObs> observations, 
                                   Map map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weight_sum = 0.0;
  int lm_size = map_landmarks.landmark_list.size();
  int obs_size = observations.size();

  for (int i=0; i<num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // set 'predictions' as landmarks within sensor range
    vector<LandmarkObs> predictions;
    for (int j=0; j<lm_size; j++) {
      int lm_id = map_landmarks.landmark_list[j].id_i;
      double lm_x = map_landmarks.landmark_list[j].x_f;
      double lm_y = map_landmarks.landmark_list[j].y_f;

      float distance = dist(p_x, p_y, lm_x, lm_y);
      if (distance <= sensor_range)
        predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
    }

    // homogeneous transform of obsevations from vehicle coord. to map coord.
    vector<LandmarkObs> observations_transformed;
    for (int j=0; j<obs_size; j++) {
      double t_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double t_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      observations_transformed.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }

    // match landmarks and transformed observations and assign id's to observations
    ParticleFilter::dataAssociation(predictions, observations_transformed);

    // store transformed observations and its associations into particle object
    // vector<int> associations;
    // vector<double> sense_x;
    // vector<double> sense_y;
    // for (int j=0; j<obs_size; j++) {
    //   associations.push_back(observations_transformed[j].id);
    //   sense_x.push_back(observations_transformed[j].x);
    //   sense_y.push_back(observations_transformed[j].y);
    // }
    // ParticleFilter::SetAssociations(particles[i], associations, sense_x, sense_y);

    // re-initalize weights
    particles[i].weight = 1.0;

    // weights as multivariate Gaussian w.r.t. landmarks
    for (int j=0; j<obs_size; j++) {
      int pred_size = predictions.size();
      for (int k=0; k<pred_size; k++) {
        if (predictions[k].id == observations_transformed[j].id)
          particles[i].weight *= multiv_prob(std_landmark[0], std_landmark[1],
                                             observations_transformed[j].x, observations_transformed[j].y,
                                             predictions[k].x, predictions[k].y);
      }
    }
    weight_sum += particles[i].weight;
  }

  // normalize weights
  for (int i=0; i<num_particles; i++)
    particles[i].weight /= weight_sum;
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // resample particles with resampling wheel
  std::default_random_engine gen;
  std::vector<Particle> particles_new(num_particles);

  uniform_int_distribution<int> dist_idx(0, num_particles-1);
  int idx = dist_idx(gen);
  double beta = 0.0;
  double max_weight = 0.0;
  for (int i=0; i<num_particles; i++)
    if (particles[i].weight > max_weight)
      max_weight = particles[i].weight;
  
  uniform_real_distribution<double> dist_beta(0.0, 2*max_weight);
  for (int i=0; i<num_particles; i++) {
    beta += dist_beta(gen);
    while (beta > particles[idx].weight) {
      beta -= particles[idx].weight;
      idx = (idx+1) % num_particles;
    }
    particles_new[i] = particles[idx];
  }
  particles = particles_new;
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
  particle.associations = associations;
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