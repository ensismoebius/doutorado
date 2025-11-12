/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 29 de mar de 2020
 *
 * @see http://wavelets.pybytes.com/
 *
 */

#ifndef SRC_LIB_WAVELET_TYPES_H_
#define SRC_LIB_WAVELET_TYPES_H_

#include <map>
#include <string>
#include <vector>
namespace wavelets
{

void init(const std::vector<std::string>& chosenWavelets = {});
void resetInitialization();
auto get(const std::string& waveletName) -> std::vector<double>;
auto all() -> std::map<std::string, std::vector<double>>;
} // namespace wavelets

#endif /* SRC_LIB_WAVELET_TYPES_H_ */
