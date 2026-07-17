/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 3 de jul de 2020
 *
 */
#ifndef SRC_LIB_STATISTICS_STATISTICS_H_
#define SRC_LIB_STATISTICS_STATISTICS_H_

#include <vector>

namespace statistics
{
/**
 * @file statistics.hpp
 * @brief Small numeric helpers (variance, standard deviation).
 *
 * These are generic utilities used by experiments/analysis code.
 * All functions operate on raw doubles and do not depend on the NN/SNN modules.
 */
auto variance(const std::vector<double>& data) -> double;
auto variance(const double* data, unsigned int length) -> double;

auto standardDeviation(const std::vector<double>& data) -> double;
auto standardDeviation(const double* data, unsigned int length) -> double;
} // namespace statistics
#endif /* SRC_LIB_STATISTICS_STATISTICS_H_ */
