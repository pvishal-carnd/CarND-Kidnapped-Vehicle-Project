/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

const double EPS = 0.0001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    // Set the number of particles
    num_particles = 100;
    weights.resize(num_particles, 1.0);
    particles.resize(num_particles);

    // Initialize Gaussian distributions around the given means and stds
    normal_distribution<double> distx(x, std[0]);
    normal_distribution<double> disty(y, std[1]);
    normal_distribution<double> distTheta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {

        // Initialize each particle with a state from the Gaussian
        Particle particle;
        particle.id = i;
        particle.x = distx(gen);
        particle.y = disty(gen);
        particle.theta = distTheta(gen);
        particle.weight = 1.0;

        particles[i] = (particle);

    }

    is_initialized = true;

    return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    // http://www.cplusplus.com/reference/random/default_random_engine/

    for (auto &p: particles){

        // Choose the right motion model depending on the yaw rate and propogate each particle according
        // to that model
        if (fabs(yaw_rate) > EPS) {
            p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y += velocity/yaw_rate * (cos(p.theta)  - cos(p.theta + yaw_rate * delta_t));
        }
        else {
            p.x += velocity * delta_t * cos(p.theta);
            p.y += velocity * delta_t * sin(p.theta);
        }
        p.theta  += yaw_rate * delta_t;
        p.theta = normalize_angle(p.theta);

        std::normal_distribution<double> distx(p.x, std_pos[0]);
        std::normal_distribution<double> disty(p.y, std_pos[1]);
        std::normal_distribution<double> distTheta(p.theta, std_pos[2]);

        p.x = distx(gen);
        p.y = disty(gen);
        p.theta = distTheta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks, std::vector<LandmarkObs>& observations) {
    // Find the landmarks closest to each observed measurement. Label each observation id with the index of the
    // landmark it is closest to

    double minimum_value;
    int    minimum_idx;

    for (int j = 0; j < observations.size(); j++){
        auto obs = observations[j];
        minimum_value = INFINITY;

        // For each observation, loop through all the landmarks and find the closest one
        //for (auto l: landmarks){
        for (int i = 0; i < landmarks.size(); i++){
            auto lm = landmarks[i];
            auto distance = distsquare(obs.x,obs.y,lm.x,lm.y);
            if (distance < minimum_value) {
                minimum_value = distance;
                minimum_idx    = i;
            }
        }

        observations[j].id = minimum_idx;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    // more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    // according to the MAP'S coordinate system. You will need to transform between the two systems.
    // Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    // The following is a good resource for the theory:
    // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    // and the following is a good resource for the actual equation to implement (look at equation
    // 3.33
    // http://planning.cs.uiuc.edu/node99.html

    for(int i=0; i < particles.size(); ++i) {

        // collect all landmarks within sensor range of the current particle in a vector predicted.
        Particle p = particles[i];

        // Transform observations from the particle frame to map frame
        std::vector<LandmarkObs> observations_global;
        for (auto obs: observations){

            LandmarkObs gobs;
            gobs.x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
            gobs.y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
            gobs.id = obs.id;

            observations_global.push_back(gobs);
        }

        // Loop through all landmarks in the map and choose the ones within the sensor range
        std::vector<LandmarkObs> landmarks_visible;
        for (auto landmark: map_landmarks.landmark_list){

            double distance = distsquare(p.x,p.y,landmark.x_f,landmark.y_f);
            distance = sqrt(distance);
            if (distance < sensor_range) {
                LandmarkObs lmo;
                lmo.id = landmark.id_i;
                lmo.x = landmark.x_f;
                lmo.y = landmark.y_f;
                landmarks_visible.push_back(lmo);
            }
        }

        // Assosiate each of the observations with landmarks
        dataAssociation(landmarks_visible, observations_global);

        double likelihood = 1;
        for (auto obs: observations_global){
            auto lm = landmarks_visible[obs.id];
            likelihood *= gausian2D( (obs.x - lm.x), std_landmark[0], (obs.y - lm.y), std_landmark[1]);
        }

        p.weight = likelihood;
        weights[i] = likelihood;

    }

    return;
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::discrete_distribution<int> d(weights.begin(), weights.end());
    std::vector<Particle> new_particles(num_particles);

    for(int i = 0; i < num_particles; ++i){
        int j = d(gen);
        new_particles[i] = particles[j];
    }

    particles = std::move(new_particles);

    return;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
        const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
