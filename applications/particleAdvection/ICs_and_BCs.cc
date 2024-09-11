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

  // The initial condition is a set of overlapping circles/spheres defined
  // by a hyperbolic tangent function. The center of each circle/sphere is
  // given by "center" and its radius is given by "radius".

  double phi = 0.0;

  if (zalesak)
    {
      // Zalesak's disk level-set
      double             radius       = 15.0;
      double             notch_width  = 5.0;
      double             notch_height = 25.0;
      dealii::Point<dim> center(50.0, 75.0);

      dealii::Functions::SignedDistance::ZalesakDisk<dim> zalesak_disk(center,
                                                                       radius,
                                                                       notch_width,
                                                                       notch_height);

      double distance = zalesak_disk.value(p);

      phi = 0.5 * (1.0 - std::tanh(distance / (std::sqrt(2) * W)));
    }
  else
    {
      // Sphere level-set
      double             radius = 15.0;
      dealii::Point<dim> center(20.0, 20.0);

      dealii::Functions::SignedDistance::Sphere<dim> sphere(center, radius);

      double distance = sphere.value(p);

      phi = 0.5 * (1.0 - std::tanh(distance / (std::sqrt(2) * W)));
    }

  if (index == 0 || index == 1 || index == 2 || index == 3 || index == 4)
    {
      scalar_IC = phi;
    }

  else
    {
      scalar_IC = 0.0;
    }

  // ---------------------------------------------------------------------
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
