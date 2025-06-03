// src/consensus_cluster.cpp
// Cross-platform consensus clustering implementation
// Compatible with both GNU GCC and Apple Clang

// #include <Rcpp.h>
#include <RcppArmadillo.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Cross-platform compatible distance calculation functions
inline double manhattan_distance(const rowvec& x, const rowvec& y) {
    return arma::sum(arma::abs(x - y));
}

inline double bray_curtis_distance(const rowvec& x, const rowvec& y) {
    double numerator = arma::sum(arma::abs(x - y));
    double denominator = arma::sum(x + y);
    return (denominator < std::numeric_limits<double>::epsilon()) ? 0.0 : numerator / denominator;
}

inline double jaccard_distance(const rowvec& x, const rowvec& y) {
    // For binary data or treating as presence/absence
    arma::uvec both_present = (x > 0) && (y > 0);
    arma::uvec any_present = (x > 0) || (y > 0);
    
    double intersection = static_cast<double>(arma::sum(both_present));
    double union_size = static_cast<double>(arma::sum(any_present));
    
    return (union_size < std::numeric_limits<double>::epsilon()) ? 0.0 : 1.0 - (intersection / union_size);
}

inline double fisher_distance(const rowvec& x, const rowvec& y) {
    // Fisher distance (based on Fisher information)
    double sum_sqrt = 0.0;
    const double epsilon = std::numeric_limits<double>::epsilon();
    
    for (arma::uword i = 0; i < x.n_elem; i++) {
        if (x(i) > epsilon && y(i) > epsilon) {
            sum_sqrt += std::sqrt(x(i) * y(i));
        }
    }
    return -2.0 * std::log(sum_sqrt + epsilon);
}

// Cross-platform distance matrix calculation
arma::mat calculate_distance_matrix(const arma::mat& data, const std::string& method) {
    arma::uword n = data.n_rows;
    arma::mat dist_matrix(n, n, arma::fill::zeros);
    
    for (arma::uword i = 0; i < n; i++) {
        for (arma::uword j = i + 1; j < n; j++) {
            double dist = 0.0;
            
            if (method == "manhattan") {
                dist = manhattan_distance(data.row(i), data.row(j));
            } else if (method == "bray_curtis") {
                dist = bray_curtis_distance(data.row(i), data.row(j));
            } else if (method == "jaccard") {
                dist = jaccard_distance(data.row(i), data.row(j));
            } else if (method == "fisher") {
                dist = fisher_distance(data.row(i), data.row(j));
            }
            
            dist_matrix(i, j) = dist;
            dist_matrix(j, i) = dist;
        }
    }
    
    return dist_matrix;
}

// Cross-platform K-means clustering implementation
arma::uvec kmeans_cluster(const arma::mat& data, int k, const std::string& distance_method) {
    arma::uword n = data.n_rows;
    arma::uword p = data.n_cols;
    
    // Initialize centroids randomly using Armadillo's random functions
    arma::mat centroids(k, p);
    arma::uvec indices = arma::randperm(n, k);
    for (int i = 0; i < k; i++) {
        centroids.row(i) = data.row(indices(i));
    }
    
    arma::uvec clusters(n);
    arma::uvec old_clusters(n, arma::fill::ones);
    old_clusters *= (k + 1); // Initialize with impossible cluster values
    
    const int max_iter = 100;
    int iter = 0;
    
    while (iter < max_iter && !arma::all(clusters == old_clusters)) {
        old_clusters = clusters;
        
        // Assign points to closest centroid
        for (arma::uword i = 0; i < n; i++) {
            double min_dist = std::numeric_limits<double>::infinity();
            int best_cluster = 0;
            
            for (int j = 0; j < k; j++) {
                double dist = 0.0;
                if (distance_method == "manhattan") {
                    dist = manhattan_distance(data.row(i), centroids.row(j));
                } else if (distance_method == "bray_curtis") {
                    dist = bray_curtis_distance(data.row(i), centroids.row(j));
                } else if (distance_method == "jaccard") {
                    dist = jaccard_distance(data.row(i), centroids.row(j));
                } else if (distance_method == "fisher") {
                    dist = fisher_distance(data.row(i), centroids.row(j));
                } else {
                    dist = manhattan_distance(data.row(i), centroids.row(j));
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            clusters(i) = static_cast<arma::uword>(best_cluster);
        }
        
        // Update centroids
        for (int j = 0; j < k; j++) {
            arma::uvec cluster_members = arma::find(clusters == static_cast<arma::uword>(j));
            if (cluster_members.n_elem > 0) {
                centroids.row(j) = arma::mean(data.rows(cluster_members), 0);
            }
        }
        
        iter++;
    }
    
    return clusters + 1; // Convert to 1-based indexing for R
}

// Main consensus clustering function - cross-platform compatible
// [[Rcpp::export]]
List consensus_cluster_cpp(const arma::mat& data,
                          const IntegerVector& k_range,
                          int iterations,
                          const std::string& distance_method,
                          double subsample_ratio,
                          double feature_ratio) {
    
    arma::uword n = data.n_rows;
    arma::uword p = data.n_cols;
    arma::uword n_subsample = std::max(2.0, std::floor(static_cast<double>(n) * subsample_ratio));
    arma::uword p_subsample = std::max(1.0, std::floor(static_cast<double>(p) * feature_ratio));
    
    List consensus_matrices(k_range.size());
    List cluster_assignments(k_range.size());
    
    for (int k_idx = 0; k_idx < k_range.size(); k_idx++) {
        int k = k_range[k_idx];
        
        arma::mat consensus_matrix(n, n, arma::fill::zeros);
        arma::mat indicator_matrix(n, n, arma::fill::zeros);
        
        for (int iter = 0; iter < iterations; iter++) {
            // Subsample observations using Armadillo's randperm
            arma::uvec sample_indices = arma::randperm(n, n_subsample);
            
            // Subsample features
            arma::uvec feature_indices = arma::randperm(p, p_subsample);
            
            // Extract subsampled data
            arma::mat sub_data = data.submat(sample_indices, feature_indices);
            
            // Perform clustering
            arma::uvec clusters = kmeans_cluster(sub_data, k, distance_method);
            
            // Update consensus and indicator matrices
            for (arma::uword i = 0; i < n_subsample; i++) {
                for (arma::uword j = i + 1; j < n_subsample; j++) {
                    arma::uword orig_i = sample_indices(i);
                    arma::uword orig_j = sample_indices(j);
                    
                    indicator_matrix(orig_i, orig_j) += 1.0;
                    indicator_matrix(orig_j, orig_i) += 1.0;
                    
                    if (clusters(i) == clusters(j)) {
                        consensus_matrix(orig_i, orig_j) += 1.0;
                        consensus_matrix(orig_j, orig_i) += 1.0;
                    }
                }
            }
        }
        
        // Normalize consensus matrix
        const double epsilon = std::numeric_limits<double>::epsilon();
        for (arma::uword i = 0; i < n; i++) {
            for (arma::uword j = 0; j < n; j++) {
                if (i != j && indicator_matrix(i, j) > epsilon) {
                    consensus_matrix(i, j) /= indicator_matrix(i, j);
                } else if (i == j) {
                    consensus_matrix(i, j) = 1.0;
                }
            }
        }
        
        // Final clustering based on consensus matrix
        arma::uvec final_clusters = kmeans_cluster(consensus_matrix, k, "manhattan");
        
        consensus_matrices[k_idx] = consensus_matrix;
        cluster_assignments[k_idx] = final_clusters;
    }
    
    return List::create(
        Named("consensus_matrices") = consensus_matrices,
        Named("cluster_assignments") = cluster_assignments
    );
}
