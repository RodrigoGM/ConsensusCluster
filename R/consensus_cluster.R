# ==============================================================================
# R/consensus_cluster.R
# ==============================================================================

#' Iterative Consensus Clustering
#'
#' @param data A numeric matrix or data frame where rows are observations and columns are features
#' @param k_range A numeric vector specifying the range of k values to test (default: 2:50)
#' @param iterations Number of iterations for consensus clustering (default: 1000)
#' @param distance_method Distance method to use: "manhattan", "bray_curtis", "jaccard", "fisher" (default: "manhattan")
#' @param subsample_ratio Proportion of samples to use in each iteration (default: 0.8)
#' @param feature_ratio Proportion of features to use in each iteration (default: 1.0)
#' @param seed Random seed for reproducibility (default: NULL)
#'
#' @return A list with elements named by k values, each containing:
#'   \item{consensus_matrix}{Consensus matrix showing co-clustering frequencies}
#'   \item{cluster_assignment}{Final cluster assignments}
#'   \item{consensus_tree}{Hierarchical clustering tree from consensus matrix}
#'
#' @examples
#' # Generate example data
#' set.seed(123)
#' data <- matrix(rnorm(100 * 20), nrow = 100, ncol = 20)
#' 
#' # Run consensus clustering
#' result <- consensus_cluster(data, k_range = 2:5, iterations = 50)
#' 
#' # Access results for k=3
#' print(result[["3"]]$cluster_assignment)
#'
#' @importFrom stats as.dist as.hclust
#' @importFrom cluster agnes
#' @importFrom ape as.phylo
#' 
#' @export
consensus_cluster <- function(data, 
                             k_range = 2:50, 
                             iterations = 1000,
                             distance_method = "manhattan",
                             subsample_ratio = 0.8,
                             feature_ratio = 1.0,
                             seed = NULL) {
  
  # Input validation
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("data must be a matrix or data frame")
  }
  
  if (is.data.frame(data)) {
    data <- as.matrix(data)
  }
  
  if (!is.numeric(data)) {
    stop("data must be numeric")
  }
  
  if (any(k_range < 2)) {
    stop("All k values must be >= 2")
  }
  
  if (max(k_range) > nrow(data)) {
    warning("Maximum k is larger than number of observations")
    k_range <- k_range[k_range <= nrow(data)]
  }
  
  distance_methods <- c("manhattan", "bray_curtis", "jaccard", "fisher")
  if (!distance_method %in% distance_methods) {
    stop(paste("distance_method must be one of:", paste(distance_methods, collapse = ", ")))
  }
  
  if (subsample_ratio <= 0 || subsample_ratio > 1) {
    stop("subsample_ratio must be between 0 and 1")
  }
  
  if (feature_ratio <= 0 || feature_ratio > 1) {
    stop("feature_ratio must be between 0 and 1")
  }
  
  # Set seed if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Call C++ implementation
  result <- consensus_cluster_cpp(
    data = data,
    k_range = as.integer(k_range),
    iterations = as.integer(iterations),
    distance_method = distance_method,
    subsample_ratio = subsample_ratio,
    feature_ratio = feature_ratio
  )
  
  # Process results and add consensus trees
  final_result <- list()
  
  for (i in seq_along(k_range)) {
    k <- k_range[i]
    k_name <- as.character(k)
    
    consensus_matrix <- result$consensus_matrices[[i]]
    cluster_assignment <- result$cluster_assignments[[i]]
    
    # Create consensus tree using hierarchical clustering
    consensus_dist <- as.dist(1 - consensus_matrix)
    consensus_tree <- cluster::agnes(consensus_dist, method = "average")
    consensus_tree <- ape::as.phylo(as.hclust(consensus_tree))
    
    final_result[[k_name]] <- list(
      consensus_matrix = consensus_matrix,
      cluster_assignment = cluster_assignment,
      consensus_tree = consensus_tree
    )
  }
  
  class(final_result) <- "consensus_cluster"
  return(final_result)
} ## end consensus_cluster



#' Print method for consensus_cluster objects
#' @param x A consensus_cluster object
#' @param ... Additional arguments (unused)
#' @export
print.consensus_cluster <- function(x, ...) {
  cat("Consensus Clustering Results\n")
  cat("============================\n")
  cat("K values tested:", names(x), "\n")
  cat("Number of observations:", nrow(x[[1]]$consensus_matrix), "\n\n")
  
  for (k_name in names(x)) {
    cat("K =", k_name, ":\n")
    cat("  Cluster sizes:", table(x[[k_name]]$cluster_assignment), "\n")
  }
} ## end print.consensus_cluster



#' Plot method for consensus_cluster objects
#' 
#' @param x A consensus_cluster object
#' @param k Which k value to plot (default: first k value)
#' @param type Type of plot: "matrix", "tree", or "both" (default: "both")
#' @param ... Additional arguments passed to plotting functions
#'
#' @importFrom ape plot.phylo
#' @importFrom grDevices heat.colors
#' @importFrom graphics image par
#' 
#' @export
plot.consensus_cluster <- function(x, k = NULL, type = "both", ...) {
  if (is.null(k)) {
    k <- names(x)[1]
  }
  
  k <- as.character(k)
  
  if (!k %in% names(x)) {
    stop("k value not found in results")
  }
  
  if (type %in% c("matrix", "both")) {
    # Plot consensus matrix as heatmap
    consensus_matrix <- x[[k]]$consensus_matrix
    image(1:nrow(consensus_matrix), 1:ncol(consensus_matrix), 
          consensus_matrix, 
          col = heat.colors(100),
          xlab = "Sample", ylab = "Sample",
          main = paste("Consensus Matrix (k =", k, ")"),
          ...)
  }
  
  if (type %in% c("tree", "both")) {
    # Plot consensus tree
    if (type == "both") {
      par(mfrow = c(1, 2))
    }
    
    ape::plot.phylo(x[[k]]$consensus_tree, 
         main = paste("Consensus Tree (k =", k, ")"),
         ...)
    
    if (type == "both") {
      par(mfrow = c(1, 1))
    }
  }
}  ## end plot.consensus_cluster
