// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/core/conditional_ostreams.h>
#include <prismspf/core/field_attributes.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief A base class for the various subcontainers of the user inputs.
 */
class ParameterBase
{
public:
  /**
   * @brief Constructor.
   */
  ParameterBase() = default;

  /**
   * @brief Virtual destructor.
   */
  virtual ~ParameterBase() = default;

  /**
   * @brief Copy constructor.
   */
  ParameterBase(const ParameterBase &parameter_base) = default;

  /**
   * @brief Copy assignment.
   */
  ParameterBase &
  operator=(const ParameterBase &parameter_base) = default;

  /**
   * @brief Move constructor.
   */
  ParameterBase(ParameterBase &&parameter_base) noexcept = default;

  /**
   * @brief Move assignment.
   */
  ParameterBase &
  operator=(ParameterBase &&parameter_base) noexcept = default;

  /**
   * @brief Postprocess parameters.
   */
  virtual void
  postprocess(const std::vector<FieldAttributes> &field_attributes) = 0;

  /**
   * @brief Validate parameters.
   */
  virtual void
  validate(const std::vector<FieldAttributes> &field_attributes) const = 0;

  /**
   * @brief Print parameters to summary.log
   */
  virtual void
  print_parameter_summary() const
  {
    ConditionalOStreams::pout_summary()
      << "================================================\n"
      << "  Base Parameter Class - Nothing Defined\n"
      << "================================================\n"
      << std::flush;
  };
};

PRISMS_PF_END_NAMESPACE
