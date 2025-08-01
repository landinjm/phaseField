// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

// ===========================================================================
// FUNCTION FOR INITIAL CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
CustomPDE<dim, degree>::setInitialCondition([[maybe_unused]] const Point<dim>  &p,
                                            [[maybe_unused]] const unsigned int index,
                                            [[maybe_unused]] number            &scalar_IC,
                                            [[maybe_unused]] Vector<double>    &vector_IC)
{
  // ---------------------------------------------------------------------
  // ENTER THE INITIAL CONDITIONS HERE
  // ---------------------------------------------------------------------
  // Enter the function describing conditions for the fields at point "p".
  // Use "if" statements to set the initial condition for each variable
  // according to its variable index

  // The initial condition is a set of overlapping circles/spheres defined
  // by a hyperbolic tangent function. The center of each circle/sphere is
  // given by "center_list" and its radius is given by "radius".

  if (index < 5)
    {
      std::vector<Point<dim>> center_list;

      // The big grains
      {
        Point<dim> center(0.2, 0.15);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.25, 0.7);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.5, 0.5);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.6, 0.85);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.85, 0.35);
        center_list.push_back(center);
      }

      // The medium grains
      {
        Point<dim> center(0.08, 0.92);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.75, 0.6);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.75, 0.1);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.2, 0.45);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.85, 0.85);
        center_list.push_back(center);
      }

      // The small grains
      {
        Point<dim> center(0.55, 0.05);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.1, 0.35);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.95, 0.65);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.9, 0.15);
        center_list.push_back(center);
      }
      {
        Point<dim> center(0.45, 0.25);
        center_list.push_back(center);
      }

      std::vector<double> rad = {0.14,
                                 0.14,
                                 0.14,
                                 0.14,
                                 0.14,
                                 0.08,
                                 0.08,
                                 0.08,
                                 0.08,
                                 0.08,
                                 0.05,
                                 0.05,
                                 0.05,
                                 0.05,
                                 0.05};

      double dist = 0.0;
      scalar_IC   = 0;

      for (unsigned int dir = 0; dir < dim; dir++)
        {
          dist += (p[dir] - center_list[index][dir] * userInputs.size[dir]) *
                  (p[dir] - center_list[index][dir] * userInputs.size[dir]);
        }
      dist = std::sqrt(dist);

      scalar_IC +=
        0.5 * (1.0 - std::tanh((dist - rad[index] * userInputs.size[0]) / 0.5));

      dist = 0.0;
      for (unsigned int dir = 0; dir < dim; dir++)
        {
          dist += (p[dir] - center_list[index + 5][dir] * userInputs.size[dir]) *
                  (p[dir] - center_list[index + 5][dir] * userInputs.size[dir]);
        }
      dist = std::sqrt(dist);

      scalar_IC +=
        0.5 * (1.0 - std::tanh((dist - rad[index + 5] * userInputs.size[0]) / 0.5));

      dist = 0.0;
      for (unsigned int dir = 0; dir < dim; dir++)
        {
          dist += (p[dir] - center_list[index + 10][dir] * userInputs.size[dir]) *
                  (p[dir] - center_list[index + 10][dir] * userInputs.size[dir]);
        }
      dist = std::sqrt(dist);

      scalar_IC +=
        0.5 * (1.0 - std::tanh((dist - rad[index + 10] * userInputs.size[0]) / 0.5));
    }
  else
    {
      scalar_IC = 0.0;
    }

  // --------------------------------------------------------------------------
}

// ===========================================================================
// FUNCTION FOR NON-UNIFORM Dirichlet BOUNDARY CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
CustomPDE<dim, degree>::setNonUniformDirichletBCs(
  [[maybe_unused]] const Point<dim>  &p,
  [[maybe_unused]] const unsigned int index,
  [[maybe_unused]] const unsigned int direction,
  [[maybe_unused]] const number       time,
  [[maybe_unused]] number            &scalar_BC,
  [[maybe_unused]] Vector<double>    &vector_BC)
{
  // --------------------------------------------------------------------------
  // ENTER THE NON-UNIFORM Dirichlet BOUNDARY CONDITIONS HERE
  // --------------------------------------------------------------------------
  // Enter the function describing conditions for the fields at point "p".
  // Use "if" statements to set the boundary condition for each variable
  // according to its variable index. This function can be left blank if there
  // are no non-uniform Dirichlet boundary conditions. For BCs that change in
  // time, you can access the current time through the variable "time". The
  // boundary index can be accessed via the variable "direction", which starts
  // at zero and uses the same order as the BC specification in parameters.in
  // (i.e. left = 0, right = 1, bottom = 2, top = 3, front = 4, back = 5).

  // -------------------------------------------------------------------------
}