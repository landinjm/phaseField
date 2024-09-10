#include "../../include/matrixFreePDE.h"

template <int dim, int degree>
class customPDE : public MatrixFreePDE<dim, degree>
{
public:
  // Constructor
  customPDE(userInputParameters<dim> _userInputs)
    : MatrixFreePDE<dim, degree>(_userInputs)
    , userInputs(_userInputs) {};

  // Function to set the initial conditions (in ICs_and_BCs.h)
  void
  setInitialCondition(const dealii::Point<dim> &p,
                      const unsigned int        index,
                      double                   &scalar_IC,
                      dealii::Vector<double>   &vector_IC) override;

  // Function to set the non-uniform Dirichlet boundary conditions (in
  // ICs_and_BCs.h)
  void
  setNonUniformDirichletBCs(const dealii::Point<dim> &p,
                            const unsigned int        index,
                            const unsigned int        direction,
                            const double              time,
                            double                   &scalar_BC,
                            dealii::Vector<double>   &vector_BC) override;

private:
#include "../../include/typeDefs.h"

  const userInputParameters<dim> userInputs;

  // Function to set the RHS of the governing equations for explicit time
  // dependent equations (in equations.cc)
  void
  explicitEquationRHS(
    variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
    dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc,
    dealii::VectorizedArray<double> element_volume) const override;

  // Function to set the RHS of the governing equations for all other equations
  // (in equations.cc)
  void
  nonExplicitEquationRHS(
    variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
    dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc,
    dealii::VectorizedArray<double> element_volume) const override;

  // Function to set the LHS of the governing equations (in equations.cc)
  void
  equationLHS(
    variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
    dealii::Point<dim, dealii::VectorizedArray<double>>              q_point_loc,
    dealii::VectorizedArray<double> element_volume) const override;

// Function to set postprocessing expressions (in postprocess.h)
#ifdef POSTPROCESS_FILE_EXISTS
  void
  postProcessedFields(
    const variableContainer<dim, degree, dealii::VectorizedArray<double>> &variable_list,
    variableContainer<dim, degree, dealii::VectorizedArray<double>> &pp_variable_list,
    const dealii::Point<dim, dealii::VectorizedArray<double>>        q_point_loc),
    dealii::VectorizedArray<double> const element_volume override;
#endif

// Function to set the nucleation probability (in nucleation.h)
#ifdef NUCLEATION_FILE_EXISTS
  double
  getNucleationProbability(variableValueContainer variable_value,
                           double                 dV) const override;
#endif

  // ================================================================
  // Methods specific to this subclass
  // ================================================================

  // ================================================================
  // Model constants specific to this subclass
  // ================================================================

  double                 W = userInputs.get_model_constant_double("W");
  dealii::Tensor<1, dim> velocity =
    userInputs.get_model_constant_rank_1_tensor("velocity");
  bool   zalesak          = userInputs.get_model_constant_bool("zalesak");
  double angular_velocity = userInputs.get_model_constant_double("angular_velocity");
  int    bdf_n            = userInputs.get_model_constant_int("bdf_n");

  double disc_center[2] = {50.0, 50.0};

  // 1/dt
  double sdt = 1.0 / userInputs.dtValue;

  // bdf coefficients
  std::vector<std::vector<double>> bdf = {
    {1.,        1.,        0,          0,         0        },
    {2. / 3.,   4. / 3.,   -1. / 3.,   0,         0        },
    {6. / 11.,  18. / 11., -9. / 11.,  2. / 11.,  0        },
    {12. / 25., 48. / 25., -36. / 25., 16. / 25., -3. / 25.}
  };
  // ================================================================
};
