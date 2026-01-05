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
auto variance(const std::vector<double>& data) -> double;
auto variance(const double* data, unsigned int length) -> double;

auto standardDeviation(const std::vector<double>& data) -> double;
auto standardDeviation(const double* data, unsigned int length) -> double;
} // namespace statistics
#endif /* SRC_LIB_STATISTICS_STATISTICS_H_ */
