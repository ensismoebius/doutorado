/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * 01 de jun de 2021
 *
 * Contains comparison routines
 */

/**
 * @file comparison.h
 * @brief Small comparison helpers (range check).
 */

#ifndef SRC_LIB_UTILITY_COMPARISON_H_
#define SRC_LIB_UTILITY_COMPARISON_H_

/**
 * Checks if an value is between another two
 * @param val
 * @param lowerLimit
 * @param upperLimit
 * @return
 */
auto inRange(const long double& val, const long double& lowerLimit, const long double& upperLimit)
    -> bool;

#endif /* SRC_LIB_UTILITY_COMPARISON_H_ */
