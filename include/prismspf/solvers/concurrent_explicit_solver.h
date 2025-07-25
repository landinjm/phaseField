// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#pragma once

#include <prismspf/core/timer.h>

#include <prismspf/solvers/concurrent_solver.h>

#include <prismspf/config.h>

PRISMS_PF_BEGIN_NAMESPACE

/**
 * @brief This class handles the explicit solves of all explicit fields
 */
template <unsigned int dim, unsigned int degree, typename number>
class ConcurrentExplicitSolver : public ConcurrentSolver<dim, degree, number>
{
public:
  /**
   * @brief Constructor.
   */
  explicit ConcurrentExplicitSolver(const SolverContext<dim, degree> &_solver_context,
                                    Types::Index                      _solve_priority = 0)
    : ConcurrentSolver<dim, degree, number>(_solver_context,
                                            FieldSolveType::Explicit,
                                            _solve_priority) {};

  /**
   * @brief Destructor.
   */
  ~ConcurrentExplicitSolver() override = default;

  /**
   * @brief Copy constructor.
   *
   * Deleted so solver instances aren't copied.
   */
  ConcurrentExplicitSolver(const ConcurrentExplicitSolver &solver_base) = delete;

  /**
   * @brief Copy assignment.
   *
   * Deleted so solver instances aren't copied.
   */
  ConcurrentExplicitSolver &
  operator=(const ConcurrentExplicitSolver &solver_base) = delete;

  /**
   * @brief Move constructor.
   *
   * Deleted so solver instances aren't moved.
   */
  ConcurrentExplicitSolver(ConcurrentExplicitSolver &&solver_base) noexcept = delete;

  /**
   * @brief Move assignment.
   *
   * Deleted so solver instances aren't moved.
   */
  ConcurrentExplicitSolver &
  operator=(ConcurrentExplicitSolver &&solver_base) noexcept = delete;

  /**
   * @brief Initialize the solver.
   */
  void
  init() override
  {
    // Call the base class init
    this->ConcurrentSolver<dim, degree, number>::init();

    // Do nothing
  };

  /**
   * @brief Reinitialize the solver.
   */
  void
  reinit() override
  {
    // Call the base class reinit
    this->ConcurrentSolver<dim, degree, number>::reinit();

    // Do nothing
  };

  /**
   * @brief Solve for a single update step.
   */
  void
  solve() override
  {
    // Call the base class solve
    this->ConcurrentSolver<dim, degree, number>::solve();

    // If the solver is empty we can just return early.
    if (this->solver_is_empty())
      {
        return;
      }

    // Otherwise, solve
    this->solve_explicit_equations(
      [this](
        std::vector<typename SolverBase<dim, degree, number>::VectorType *>       &dst,
        const std::vector<typename SolverBase<dim, degree, number>::VectorType *> &src)
      {
        this->get_system_matrix()->compute_explicit_update(dst, src);
      });
  };

  /**
   * @brief Print information about the solver to summary.log.
   */
  void
  print() override
  {
    // Print the base class information
    this->ConcurrentSolver<dim, degree, number>::print();
  }
};

PRISMS_PF_END_NAMESPACE
