# Filename: quick.py
# Created: 15-10-2025
# Description: Implementation of the Elliptic Beam Approximation in ref.
# ref: Carlo Liorni et al 2019 New J. Phys. 21 093055

import numpy as np
from scipy.integrate import dblquad
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob
from multiprocessing import Pool, cpu_count
import functools

class elliptic_beam_model():
    """
    See Carlo Liorni et al 2019 New J. Phys. 21 093055
    """
    def __init__(self):
        pass

    def set_params(self, k, Cn2, n0, W0, h, L, a_r, beta):
        self.k = k
        self.Cn2 = Cn2
        self.n0 = n0
        self.W0 = W0
        self.h = h
        self.L = L
        self.a_r = a_r
        self.beta = beta
        self.sigma_R2 = 1.23 * Cn2 * k**(7/6) * L**(11/6)    # Rytov variance
        self.Omega = (k * W0**2) / (2 * L)                   # Fresnel number
        
        # Centroid shift (beam wandering)
        self.centroid_variance = 0.419 * self.sigma_R2 * W0**2 * self.Omega**(-7/6) * (h/L)
        self.centroid_std = np.sqrt(self.centroid_variance)
        # required for calculatig Θ stats
        self.beam_variance_factor = (self.W0**2 / self.Omega**2) * (1 + (np.pi/8) * (self.L) * (self.n0) * self.W0**2 * (self.h/self.L) + 2.6 * self.sigma_R2 * self.Omega**(5/6) * (self.h/self.L))
        self.variance_w_squared = 1.2 * (self.W0**4 / self.Omega**(19/6)) * (1 + (np.pi/8) * (self.L) * (self.n0) * self.W0**2 * (self.h/self.L)) * self.sigma_R2 * (self.h/self.L)
        self.cross_correlation_factor = -0.8 * (self.W0**4 / self.Omega**(19/6)) * (1 + (np.pi/8) * (self.L) * (self.n0) * self.W0**2 * (self.h/self.L)) * self.sigma_R2 * (self.h/self.L)
        
        # Θ statistics
        self.mean_theta_i = np.log(self.beam_variance_factor / self.W0**2) - 0.5 * np.log(1 + self.variance_w_squared / self.beam_variance_factor**2)
        self.variance_theta = np.log(1 + self.variance_w_squared / self.beam_variance_factor**2)
        self.covariance_theta = np.log(1 + self.cross_correlation_factor / self.beam_variance_factor**2)
        self.cov_matrix_theta = np.array([[self.variance_theta, self.covariance_theta],
                                    [self.covariance_theta, self.variance_theta]])



    def transmittance(self, x0, y0, w1, w2, theta):
        def integrand(rho, phi):
            rho0 = np.sqrt(x0**2 + y0**2)
            phi0 = np.arctan2(y0, x0) if rho0 > 0 else 0
            
            d = theta - phi0
            cos_d, sin_d = np.cos(d), np.sin(d)
            
            A = (cos_d**2)/(w1**2) + (sin_d**2)/(w2**2)
            B = (sin_d**2)/(w1**2) + (cos_d**2)/(w2**2)  
            C = (1/(w1**2) - 1/(w2**2)) * np.sin(2*d)
            
            # FIXED to account for beam center offset
            exp_term1 = np.exp(-2*A*(rho*np.cos(phi) - rho0*np.cos(phi0))**2)
            exp_term2 = np.exp(-2*B*(rho*np.sin(phi) - rho0*np.sin(phi0))**2)
            exp_term3 = np.exp(-2*C*(rho*np.cos(phi) - rho0*np.cos(phi0))*(rho*np.sin(phi) - rho0*np.sin(phi0)))
            
            return rho * exp_term1 * exp_term2 * exp_term3
        
        result, error = dblquad(integrand, 0, 2*np.pi, lambda phi: 0, lambda phi: self.a_r)
        self.chi_ext = np.exp(-1*self.beta)  # Extinction coefficient
        const = (2*self.chi_ext)/(np.pi * w1 * w2)
        return result * const, error * const
    
    def sample_beam_parameters(self):
        x0 = np.random.normal(0, self.centroid_std)
        y0 = np.random.normal(0, self.centroid_std)
        
        theta = np.random.uniform(0, np.pi/2)
        
        # Sample Θ₁ and Θ₂ and reconstruct W₁ and W₂
        theta1, theta2 = np.random.multivariate_normal([self.mean_theta_i, self.mean_theta_i], self.cov_matrix_theta)
        w1 = np.sqrt(self.W0**2 * np.exp(theta1))
        w2 = np.sqrt(self.W0**2 * np.exp(theta2))
        
        return x0, y0, w1, w2, theta

    def show_params(self):
        W0 = self.W0
        sigma_R2 = self.sigma_R2
        Omega = self.Omega
        
        centroid_variance = self.centroid_variance
        mean_theta_i = self.mean_theta_i
        variance_theta = self.variance_theta

        # NOTE: stats indirectly calculated because log normal dist was used to sample w1,w2
        mean_w_squared = W0**2 * np.exp(mean_theta_i + variance_theta/2)
        var_w_squared = W0**4 * np.exp(2*mean_theta_i + variance_theta) * (np.exp(variance_theta) - 1)
        mean_w = np.sqrt(mean_w_squared)
        var_w = var_w_squared / (4 * mean_w_squared)

        
        print(f"Rytov variance: {sigma_R2:.6e}")
        print(f"Fresnel number: {Omega:.6e}")
        print(f"x0,y0 mean: 0, variance: {centroid_variance:.6e}")
        print(f"w1,w2 mean: {mean_w:.6e}, variance: {var_w:.6e}")
    
    # Worker function for multiprocessing
    @staticmethod
    def _monte_carlo_worker(args):
        """Worker function for parallel Monte Carlo iteration"""
        model_params, beta, a_r = args
        k, Cn2, n0, W0, h, L = model_params
        
        # Create temporary model instance for this worker
        temp_model = elliptic_beam_model()
        temp_model.set_params(k, Cn2, n0, W0, h, L, a_r, beta)
        
        # Sample beam parameters
        x0, y0, w1, w2, theta = temp_model.sample_beam_parameters()
        
        # Calculate transmittance
        trans_val, _ = temp_model.transmittance(x0, y0, w1, w2, theta)
        return trans_val
    
    # Parallel Monte Carlo
    def run_monte_carlo_parallel(self, num_iterations=15000, n_processes=None):
        """
        Parallel Monte Carlo simulation using multiprocessing
        """
        if n_processes is None:
            n_processes = min(cpu_count(), 8)  # Don't use all cores to keep system responsive
        
        print(f"Starting parallel Monte Carlo with {num_iterations} iterations using {n_processes} processes...")
        
        # Prepare arguments for each worker
        model_params = (self.k, self.Cn2, self.n0, self.W0, self.h, self.L)
        worker_args = [(model_params, self.beta, self.a_r) for _ in range(num_iterations)]
        
        # Run parallel computation
        with Pool(processes=n_processes) as pool:
            transmittance_values = pool.map(self._monte_carlo_worker, worker_args)
        
        print(f"Parallel Monte Carlo simulation completed! Total iterations: {len(transmittance_values)}")
        return transmittance_values
    
    # OLD: slower Monte Carlo
    def run_monte_carlo(self, num_iterations = 15000, print_interval = None):
        if not print_interval:
            print_interval = num_iterations
        
        transmittance_values = []

        print(f"Running Monte Carlo simulation with {num_iterations} iterations...")
        print(f"Progress will be printed every {print_interval} iterations")

        for counter in range(1, num_iterations + 1):
            x0, y0, w1, w2, theta = self.sample_beam_parameters()
            trans_val, _ = self.transmittance(x0, y0, w1, w2, theta)
            transmittance_values.append(trans_val)
            
            if not (counter % print_interval):
                print(f"Completed {counter} iterations")

        print(f"Monte Carlo simulation completed! Total iterations: {len(transmittance_values)}")
        return transmittance_values
    
    def show_dist_stats(self, transmittance_values):
        transmittance_array = np.array(transmittance_values)  # Convert to numpy array
        print(f"Samples: {len(transmittance_array)}")
        print(f"Mean: {np.mean(transmittance_array):.6f}")
        print(f"Std Dev: {np.std(transmittance_array):.6f}")
        print(f"Min: {np.min(transmittance_array):.6f}")
        print(f"Max: {np.max(transmittance_array):.6f}")

    def save_data(self, transmittance_values, filena = None):
        # Save simulation data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_file_name = f'transmittance_results_{timestamp}.pkl'
        if (filena):
            saved_file_name = f'{filena}.pkl'
        data_dict = {
            'transmittance_values': transmittance_values
        }
        with open(saved_file_name, 'wb') as f:
            pickle.dump(data_dict, f)
    
    def load_data(self, filena = None):
        # Load simulation data
        def find_latest_pickle():
            pickle_files = glob.glob('*.pkl')
            if not pickle_files:
                print("No pickle files found.")
                return None
            latest_file = max(pickle_files, key=os.path.getmtime)
            print(f"Found pickle file: {latest_file}")
            return latest_file
        if not(filena):
            pickle_file = find_latest_pickle()
        if (filena):
            pickle_file = filena + ".pkl"
        if pickle_file:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
                return data['transmittance_values']
        else:
            print("Could not find pickle file. Please check the filename.")
    
    def _savefig(self, plot_type, figname = None):
        if figname:
            figname = f'{figname}.png'
            plt.savefig(figname, dpi=300, bbox_inches='tight')
            print(f"Saved fig as : {figname}")

    def plot_my_hist(self, data, plot_type = "pdf", figname = None):
        transmittance_array = np.array(data)
        data_range = np.max(transmittance_array) - np.min(transmittance_array)
        print(f"Data range: {data_range:.6f}")

        iqr = np.percentile(transmittance_array, 75) - np.percentile(transmittance_array, 25)
        fd_bin_width = 2 * iqr * (len(transmittance_array) ** (-1/3))
        n_bins = max(10, min(100, int(data_range / fd_bin_width)))

        print(f"Using {n_bins} bins for histogram.")
        
        # Statistics
        stats_text = f'Mean: {np.mean(transmittance_array):.6f}\n'
        stats_text += f'Std Dev: {np.std(transmittance_array):.6f}\n'
        stats_text += f'Min: {np.min(transmittance_array):.6f}\n'
        stats_text += f'Max: {np.max(transmittance_array):.6f}'

        plt.figure(figsize=(12, 8))

        if plot_type == "count":
            count_alt, bins_alt, _ = plt.hist(transmittance_array, bins=n_bins, 
                                                        density=False, alpha=0.7, color='lightcoral', 
                                                        edgecolor='black', linewidth=0.5)
            
            plt.xlabel('Transmittance', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title(f'PDT - Count Histogram (N={len(transmittance_array)})', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add statistics text box
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            self._savefig(plot_type, figname)
            plt.show()
        elif plot_type == "pdf":
            count, bins, _ = plt.hist(transmittance_array, bins=n_bins, alpha=0)
            plt.close()
            total_items = np.sum(count)
            normalized_counts = count / total_items
            plt.figure(figsize=(12, 8))
            plt.bar(bins[:-1], normalized_counts, width=bins[1]-bins[0], 
                    alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)

            plt.xlabel('Transmittance')
            plt.ylabel('Probability')
            plt.title(f'PDT (N={len(transmittance_array)})')
            plt.grid(True, alpha=0.3)
            self._savefig(plot_type, figname)
            plt.show()
