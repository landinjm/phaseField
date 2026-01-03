// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/core/conditional_ostreams.h>
#include <prismspf/core/field_attributes.h>

#include <prismspf/user_inputs/parameter_base.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
// cppcoreguidelines-non-private-member-variables-in-classes)

class TemporalParameters : public ParameterBase
{
public:
  /**
   * @brief Constructor.
   */
  explicit TemporalParameters(double       _initial_timestep = 0.0,
                              unsigned int _total_increments = 0,
                              double       _initial_time     = 0.0,
                              double       _final_time       = 0.0)
    : initial_timestep(_initial_timestep)
    , total_increments(_total_increments)
    , initial_time(_initial_time)
    , final_time(_final_time)
  {}

  /**
   * @brief Postprocess parameters.
   */
  void
  postprocess(const std::vector<FieldAttributes> &field_attributes) override
  {}

  /**
   * @brief Validate parameters.
   */
  void
  validate(const std::vector<FieldAttributes> &field_attributes) const override
  {}

  /**
   * @brief Print parameters to summary.log
   */
  void
  print_parameter_summary() const override
  {
    ConditionalOStreams::pout_summary()
      << "================================================\n"
      << "  Temporal Parameters\n"
      << "================================================\n"
      << "Initial timestep: " << initial_timestep << "\n"
      << "Total increments: " << total_increments << "\n"
      << "Initial time: " << initial_time << "\n"
      << "Final time: " << final_time << "\n\n"
      << std::flush;
  }

  /**
   * @brief Initial timestep.
   */
  double initial_timestep = 0.0;

  /**
   * @brief Total number of increments.
   */
  unsigned int total_increments = 0;

  /**
   * @brief Initial time.
   */
  double initial_time = 0.0;

  /**
   * @brief Final time.
   */
  double final_time = 0.0;

private:
  /**
   * @brief Whether we only have time independent fields
   */
  bool only_time_independent_fields = true;
};

// NOLINTEND(misc-non-private-member-variables-in-classes,
// cppcoreguidelines-non-private-member-variables-in-classes)

PRISMS_PF_END_NAMESPACE
