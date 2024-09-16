// ===========================================================================
// FUNCTION FOR INITIAL CONDITIONS
// ===========================================================================
#include <deal.II/base/function_signed_distance.h>

template <int dim, int degree>
void
customPDE<dim, degree>::setInitialCondition(const dealii::Point<dim> &p,
                                            const unsigned int        index,
                                            double                   &scalar_IC,
                                            dealii::Vector<double>   &vector_IC)
{
  // ---------------------------------------------------------------------
  // ENTER THE INITIAL CONDITIONS HERE
  // ---------------------------------------------------------------------
  // Enter the function describing conditions for the fields at point "p".
  // Use "if" statements to set the initial condition for each variable
  // according to its variable index

  // The initial condition is two circles/spheres defined
  // by a hyperbolic tangent function. The center of each circle/sphere is
  // given by "center" and its radius is given by "rad".

  // Sphere level-set
  double             phi    = 0.0;
  double             radius = 5.0;
  dealii::Point<dim> center(0.0, 0.0);

  dealii::Functions::SignedDistance::Sphere<dim> sphere(center, radius);

  double distance = sphere.value(p);

  phi = -std::tanh(distance / std::sqrt(2));

  // Sphere level-set
  double             psi_level_set   = 0.0;
  double             radius_particle = 5.0;
  dealii::Point<dim> center_particle(20.0, 20.0);

  dealii::Functions::SignedDistance::Sphere<dim> particle(center_particle,
                                                          radius_particle);

  psi_level_set = particle.value(p);

  // Initial condition for the concentration field
  if (index == 0)
    {
      scalar_IC = U0;
    }
  // Initial condition for the order parameter field
  else if (index == 1)
    {
      scalar_IC = phi;
    }

  else if (index == 3 || index == 4)
    {
      scalar_IC = psi_level_set;
    }

  // --------------------------------------------------------------------------
}

// ===========================================================================
// FUNCTION FOR NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS
// ===========================================================================

template <int dim, int degree>
void
customPDE<dim, degree>::setNonUniformDirichletBCs(const dealii::Point<dim> &p,
                                                  const unsigned int        index,
                                                  const unsigned int        direction,
                                                  const double              time,
                                                  double                   &scalar_BC,
                                                  dealii::Vector<double>   &vector_BC)
{
  // --------------------------------------------------------------------------
  // ENTER THE NON-UNIFORM DIRICHLET BOUNDARY CONDITIONS HERE
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
