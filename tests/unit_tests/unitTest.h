#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>

// define data type
template <int dim>
void
computeStress(const dealii::Table<2, double>       &CIJ,
              const dealii::VectorizedArray<double> ux[][dim],
              const dealii::VectorizedArray<double> R[][dim]);

#include "FloodFiller/FloodFiller.cc"
#include "OrderParameterRemapper/OrderParameterRemapper.cc"
#include "SimplifiedGrainRepresentation/SimplifiedGrainRepresentation.cc"
#include "computeStress.h"
#include "inputFileReader/inputFileReader.cc"
#include "matrixFreePDE.h"
#include "matrixfree/AdaptiveRefinement.cc"
#include "matrixfree/boundaryConditions.cc"
#include "matrixfree/checkpoint.cc"
#include "matrixfree/computeIntegral.cc"
#include "matrixfree/computeLHS.cc"
#include "matrixfree/computeRHS.cc"
#include "matrixfree/init.cc"
#include "matrixfree/initForTests.cc"
#include "matrixfree/initialConditions.cc"
#include "matrixfree/invM.cc"
#include "matrixfree/markBoundaries.cc"
#include "matrixfree/matrixFreePDE.cc"
#include "matrixfree/nucleation.cc"
#include "matrixfree/outputResults.cc"
#include "matrixfree/postprocessor.cc"
#include "matrixfree/reassignGrains.cc"
#include "matrixfree/reinit.cc"
#include "matrixfree/setNonlinearEqInitialGuess.cc"
#include "matrixfree/solve.cc"
#include "matrixfree/solveIncrement.cc"
#include "matrixfree/utilities.cc"
#include "parallelNucleationList.h"
#include "parallelNucleationList/parallelNucleationList.cc"
#include "userInputParameters/loadVariableAttributes.cc"
#include "userInputParameters/load_BC_list.cc"
#include "userInputParameters/load_user_constants.cc"
#include "userInputParameters/setTimeStepList.cc"
#include "utilities/sortIndexEntryPairList.cc"
#include "variableAttributeLoader/variableAttributeLoader.cc"
#include "variableContainer/variableContainer.cc"

template <int dim, typename T>
class unitTest
{
public:
  bool
  test_computeInvM(int argc, char **argv, userInputParameters<dim>);
  bool
  test_outputResults(int argc, char **argv, userInputParameters<dim> userInputs);
  bool
  test_computeStress();
  void
  assignCIJSize(
    dealii::VectorizedArray<double> CIJ[2 * dim - 1 + dim / 3][2 * dim - 1 + dim / 3]);
  void
  assignCIJSize(dealii::Table<2, double> &CIJ);
  bool
  test_setRigidBodyModeConstraints(std::vector<int>, userInputParameters<dim> userInputs);
  bool
  test_parse_line();
  bool
  test_get_subsection_entry_list();
  bool
  test_get_entry_name_ending_list();
  bool
  test_load_BC_list();
  bool
  test_setOutputTimeSteps();
  bool
  test_NonlinearSolverParameters();
  bool
  test_LinearSolverParameters();
  bool
  test_EquationDependencyParser_variables_and_residuals_needed();
  bool
  test_EquationDependencyParser_nonlinear();
  bool
  test_EquationDependencyParser_postprocessing();
  bool
  test_FloodFiller();
  bool
  test_SimplifiedGrainRepresentation();
  bool
  test_SimplifiedGrainManipulator_transferGrainIds();
  bool
  test_SimplifiedGrainManipulator_reassignGrains();
  bool
  test_OrderParameterRemapper();
};

#include "EquationDependencyParser/EquationDependencyParser.cc"
#include "SolverParameters.h"
#include "SolverParameters/SolverParameters.cc"
#include "test_EquationDependencyParser.h"
#include "test_FloodFiller.h"
#include "test_LinearSolverParameters.h"
#include "test_NonlinearSolverParameters.h"
#include "test_OrderParameterRemapper.h"
#include "test_SimplifiedGrainManipulator.h"
#include "test_SimplifiedGrainRepresentation.h"
#include "test_computeStress.h"
#include "test_get_entry_name_ending_list.h"
#include "test_get_subsection_entry_list.h"
#include "test_invM.h"
#include "test_load_BC_list.h"
#include "test_outputResults.h"
#include "test_parse_line.h"
#include "test_setOutputTimeSteps.h"
#include "test_setRigidBodyModeConstraints.h"
#include "variableAttributeLoader_test.cc"
