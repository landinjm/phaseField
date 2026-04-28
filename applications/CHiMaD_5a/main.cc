// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>

using namespace prisms;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  prisms::MPIInitFinalize mpi_init(argc, argv);

  // Parse the command line options (if there are any) to get the name of the input
  // file
  ParseCMDOptions cli_options(argc, argv);

  constexpr unsigned int degree = 2;

  std::vector<FieldAttributes> fields = {FieldAttributes("u", Vector),
                                         FieldAttributes("p")};

  SolveBlock implicits;
  implicits.id               = 1;
  implicits.solve_type       = Linear;
  implicits.solve_timing     = Uninitialized;
  implicits.field_indices    = {0, 1};
  implicits.dependencies_lhs = make_dependency_set(
    fields,
    {"lhs(u)", "grad(lhs(u))", "hess(lhs(u))", "lhs(p)", "grad(lhs(p))"});

  std::vector<SolveBlock> solve_groups({implicits});

  UserInputParameters<dim>     user_inputs(cli_options.get_parameters_filename());
  PhaseFieldTools<dim>         pf_tools;
  CustomPDE<degree, double>    pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double> problem(fields,
                                       solve_groups,
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);
  problem.solve();

  return 0;
}
