// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <deal.II/base/exceptions.h>

#include <prismspf/core/types.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

// NOLINTBEGIN(readability-identifier-naming)

/**
 * Function for deal.II AssertThrow that is only valid in DEBUG mode. This is used to
 * throw and error that can be caught by catch2 while not bloating code in release mode.
 */
#ifdef DEBUG
template <typename Condition, typename Exception>
constexpr void
AssertThrowDebug(const Condition &cond, const Exception &exc)
{
  AssertThrow(cond, exc);
}
#else
template <typename Condition, typename Exception>
constexpr void
AssertThrowDebug(const Condition &, const Exception &)
{
  // Do nothing in release mode
}
#endif

// NOLINTEND(readability-identifier-naming)

// NOLINTBEGIN

/**
 * Exception for parts of the library that have yet to be implemented yet. The argument is
 * used to provide additional context for the feature that has yet to be implemented.
 */
DeclException1(
  FeatureNotImplemented,
  std::string,
  << "The following feature has yet to be implemented in PRISMS-PF:\n  " << arg1
  << "\nCheck the issues section of PRISMS-PF's github to see if this feature is under "
     "development. Additionally, please considering provided a patch to PRISMS-PF if you "
     "feel that feature is worthwhile for yourself and others.");

/**
 * Exception for when we have reached part of the code that should not have been reached.
 * This is common in switch statements and conditional chains.
 */
DeclExceptionMsg(UnreachableCode, "This code should not have been reached.");

/**
 * Exception for a user trying to access a variable in from VariableContainer that has not
 * been specified as a dependency.
 */
DeclException2(DependencyNotFound,
               Types::Index,
               std::string,
               << "Attemped access of the variable with index " << arg1
               << " and dependency type " << arg2
               << " that was not marked as needed. Please check CustomAttributeLoader.");

// NOLINTEND

PRISMS_PF_END_NAMESPACE