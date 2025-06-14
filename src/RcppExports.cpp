// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// consensus_cluster_cpp
List consensus_cluster_cpp(const arma::mat& data, const IntegerVector& k_range, int iterations, const std::string& distance_method, double subsample_ratio, double feature_ratio);
RcppExport SEXP _ConsensusCluster_consensus_cluster_cpp(SEXP dataSEXP, SEXP k_rangeSEXP, SEXP iterationsSEXP, SEXP distance_methodSEXP, SEXP subsample_ratioSEXP, SEXP feature_ratioSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< const IntegerVector& >::type k_range(k_rangeSEXP);
    Rcpp::traits::input_parameter< int >::type iterations(iterationsSEXP);
    Rcpp::traits::input_parameter< const std::string& >::type distance_method(distance_methodSEXP);
    Rcpp::traits::input_parameter< double >::type subsample_ratio(subsample_ratioSEXP);
    Rcpp::traits::input_parameter< double >::type feature_ratio(feature_ratioSEXP);
    rcpp_result_gen = Rcpp::wrap(consensus_cluster_cpp(data, k_range, iterations, distance_method, subsample_ratio, feature_ratio));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ConsensusCluster_consensus_cluster_cpp", (DL_FUNC) &_ConsensusCluster_consensus_cluster_cpp, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_ConsensusCluster(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
