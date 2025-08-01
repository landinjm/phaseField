// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/core/conditional_ostreams.h>
#include <prismspf/core/variable_attributes.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief Struct that holds temporal discretization parameters.
 */
struct TemporalDiscretization
{
public:
  /**
   * @brief Postprocess and validate parameters.
   */
  void
  postprocess_and_validate(
    const std::map<unsigned int, VariableAttributes> &var_attributes);

  /**
   * @brief Print parameters to summary.log
   */
  void
  print_parameter_summary() const;

  /**
   * @brief Get the final time.
   */
  double
  get_final_time() const
  {
    return final_time;
  }

  /**
   * @brief Set the final time.
   */
  void
  set_final_time(double _final_time)
  {
    final_time = _final_time;
  }

  /**
   * @brief Get the time.
   */
  double
  get_time() const
  {
    return time;
  }

  /**
   * @brief Update the time with the timestep.
   *
   * Note that this function is const even though it does increment the timestep.
   */
  void
  update_time() const
  {
    time += dt;
  }

  /**
   * @brief Get the total number of increments.
   */
  unsigned int
  get_total_increments() const
  {
    return total_increments;
  }

  /**
   * @brief Set the total number of increments.
   */
  void
  set_total_increments(unsigned int _total_increments)
  {
    total_increments = _total_increments;
  }

  /**
   * @brief Get the increment.
   */
  unsigned int
  get_increment() const
  {
    return increment;
  }

  /**
   * @brief Update the increment by one.
   *
   * Note that this function is const even though it does increment the increment.
   */
  void
  update_increment() const
  {
    increment++;
  }

  /**
   * @brief Get the timestep.
   */
  double
  get_timestep() const
  {
    return dt;
  }

  /**
   * @brief Set the timestep.
   */
  void
  set_timestep(double _dt)
  {
    dt = _dt;
  }

private:
  // The increment
  mutable unsigned int increment = 0;

  // Total number of increments
  unsigned int total_increments = 0;

  // Timestep
  mutable double dt = 0.0;

  // The time
  mutable double time = 0.0;

  // Final time
  double final_time = 0.0;
};

inline void
TemporalDiscretization::postprocess_and_validate(
  const std::map<unsigned int, VariableAttributes> &var_attributes)
{
  // If all of the variables are `TimeIndependent`, `Auxiliary`, or `Constant` then
  // total_increments should be 0 and final_time should be 0
  bool only_time_independent_pdes = true;
  for (const auto &[index, variable] : var_attributes)
    {
      if (variable.is_postprocess())
        {
          continue;
        }
      if (variable.get_pde_type() == PDEType::ExplicitTimeDependent ||
          variable.get_pde_type() == PDEType::ImplicitTimeDependent)
        {
          only_time_independent_pdes = false;
          break;
        }
    }
  if (only_time_independent_pdes)
    {
      total_increments = 0;
      return;
    }

  // Check that the timestep is greater than zero
  AssertThrow(dt > 0.0,
              dealii::ExcMessage(
                "The timestep must be greater than zero for transient problems."));

  // Pick the maximum specified time since the default values are zero
  final_time       = std::max(final_time, dt * total_increments);
  total_increments = static_cast<unsigned int>(std::ceil(final_time / dt));
}

inline void
TemporalDiscretization::print_parameter_summary() const
{
  ConditionalOStreams::pout_summary()
    << "================================================\n"
    << "  Temporal Discretization\n"
    << "================================================\n"
    << "Timestep: " << dt << "\n"
    << "Total increments: " << total_increments << "\n"
    << "Final time: " << final_time << "\n\n"
    << std::flush;
}

PRISMS_PF_END_NAMESPACE