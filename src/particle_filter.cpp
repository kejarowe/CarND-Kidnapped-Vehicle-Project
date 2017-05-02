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

#include "particle_filter.h"

using namespace std;

double BivariateGaussianCalc(double x_diff, double y_diff, double std_landmark[])
{
	return (1.0 / (2 * M_PI*std_landmark[0] * std_landmark[1])) * exp(-((pow(x_diff,2) / (2*pow(std_landmark[0],2))) 
		+ (pow(y_diff,2) / (2*pow(std_landmark[1],2)))));
};

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	is_initialized = true;
	num_particles = 200; 
	
	// Create Gaussian distributions for state variables.
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	Particle current_particle;

	for (int i = 0; i < num_particles; ++i) {
		current_particle.id = i;
		current_particle.x = dist_x(gen); 
		current_particle.y = dist_y(gen);
		current_particle.theta = dist_theta(gen);
		current_particle.weight = 1.0 / num_particles;

		particles.push_back(current_particle);
		weights.push_back(current_particle.weight);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	for (auto && particle : particles){
		//Model-based prediction
		if (yaw_rate == 0) {
			particle.x += cos(particle.theta) * velocity * delta_t;
			particle.y += sin(particle.theta) * velocity * delta_t;
		}
		else {
			particle.x += (velocity / yaw_rate) * (sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
			particle.y += (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
		}
		particle.theta += yaw_rate * delta_t;

		//addition of Gaussian noise
		normal_distribution<double> dist_x(particle.x, 2*std_pos[0]);
		normal_distribution<double> dist_y(particle.y, 2*std_pos[1]);
		normal_distribution<double> dist_theta(particle.theta, 2*std_pos[2]);

		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	double closest_distance, current_distance;
	for (auto && observation : observations){
		closest_distance = numeric_limits<double>::infinity();
		for (auto prediction : predicted){
			current_distance = sqrt(pow(prediction.x - observation.x,2) + pow(prediction.y - observation.y,2));
			if (current_distance < closest_distance) {
				closest_distance = current_distance;
				observation.id = prediction.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	static vector<LandmarkObs> map;
	if (map.size() == 0)
	{
		int id = 0;
		for (auto landmark : map_landmarks.landmark_list){
			LandmarkObs obs;
			obs.id = id;
			obs.x = landmark.x_f;
			obs.y = landmark.y_f;
			map.push_back(obs);
			id++;
		}
	}
	double sum_weights = 0;
	std::vector<LandmarkObs> observations_in_map_frame;
	for (auto && particle : particles) {
		observations_in_map_frame.clear();
		for (auto observation : observations){
			//convert observation to map coordinate frame
			LandmarkObs current_observation;
			current_observation.id = observation.id;
			current_observation.x = particle.x + observation.x*cos(particle.theta) - observation.y*sin(particle.theta);
			current_observation.y = particle.y + observation.x*sin(particle.theta) + observation.y*cos(particle.theta);
			observations_in_map_frame.push_back(current_observation);
		}
		dataAssociation(map,observations_in_map_frame);
		double weight = 1;
		for (auto observation : observations_in_map_frame) {
			weight *= BivariateGaussianCalc(observation.x - map[observation.id].x, observation.y - map[observation.id].y, std_landmark);
		}
		particle.weight = weight;
		sum_weights += weight; 
	}
	weights.clear();
	for (auto && particle : particles){
		particle.weight /= sum_weights;
		weights.push_back(particle.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
    discrete_distribution<double> d(weights.begin(),weights.end());

    std::vector<Particle> new_particles;
    double sum_weights = 0;
    for (int i = 0; i < num_particles; i++) {
    	int particle_index = d(gen);
    	sum_weights += particles[particle_index].weight;
    	new_particles.push_back(particles[particle_index]);
	}

    //renormalize weights
    for (auto && particle : new_particles){
    	particle.weight /= sum_weights;
    }

    //set member variable
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
