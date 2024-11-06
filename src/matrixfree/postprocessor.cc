// #include "../../include/postprocessor.h"
#include "../../include/matrixFreePDE.h"

template <int dim, int degree>
void
MatrixFreePDE<dim, degree>::computePostProcessedFields(
  std::vector<vectorType *> &postProcessedSet)
{
  // find indices of first occuring scalar & vector field
  unsigned int scalarField = 0;
  bool         foundScalar = false;
  unsigned int vectorField = 0;
  bool         foundVector = false;
  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      if (fields[fieldIndex].type == SCALAR && !foundScalar)
        {
          scalarField = fieldIndex;
          foundScalar = true;
        }
      else if (fields[fieldIndex].type == VECTOR && !foundVector)
        {
          vectorField = fieldIndex;
          foundVector = true;
        }
    }

  // Initialize the postProcessedSet
  pcout << "initializing parallel::distributed postprocessed vectors\n";
  for (unsigned int fieldIndex = 0; fieldIndex < userInputs.pp_number_of_variables;
       fieldIndex++)
    {
      vectorType *U;
      U = new vectorType;
      postProcessedSet.push_back(U);

      if (userInputs.pp_varInfoList[fieldIndex].is_scalar)
        {
          matrixFreeObject.initialize_dof_vector(*U, scalarField);
          *U = 0;
        }
      else
        {
          pcout << "Vector postprocess variables are currently not supported"
                << std::endl;
          exit(-1);
        }
    }

  integrated_postprocessed_fields.clear();
  for (unsigned int index = 0; index < userInputs.num_integrated_fields; index++)
    {
      integrated_postprocessed_fields.push_back(0.0);
    }

  // call to integrate and assemble
  matrixFreeObject.cell_loop(&MatrixFreePDE<dim, degree>::getPostProcessedFields,
                             this,
                             postProcessedSet,
                             solutionSet,
                             true);
}

template <int dim, int degree>
void
MatrixFreePDE<dim, degree>::getPostProcessedFields(
  const dealii::MatrixFree<dim, double>       &data,
  std::vector<vectorType *>                   &dst,
  const std::vector<vectorType *>             &src,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // find indices of first occuring scalar & vector field
  unsigned int scalarField = 0;
  bool         foundScalar = false;
  unsigned int vectorField = 0;
  bool         foundVector = false;
  for (unsigned int fieldIndex = 0; fieldIndex < fields.size(); fieldIndex++)
    {
      if (fields[fieldIndex].type == SCALAR && !foundScalar)
        {
          scalarField = fieldIndex;
          foundScalar = true;
        }
      else if (fields[fieldIndex].type == VECTOR && !foundVector)
        {
          vectorField = fieldIndex;
          foundVector = true;
        }
    }

  // initialize FEEvaulation objects
  variableContainer<dim, degree, dealii::VectorizedArray<double>> variable_list(
    data,
    userInputs.pp_baseVarInfoList);
  variableContainer<dim, degree, dealii::VectorizedArray<double>>
    pp_variable_list(data, userInputs.pp_varInfoList, scalarField);

  // loop over cells
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // Initialize, read DOFs, and set evaulation flags for each variable
      variable_list.reinit_and_eval(src, cell);
      pp_variable_list.reinit(cell);

      unsigned int num_q_points = variable_list.get_num_q_points();

      dealii::VectorizedArray<double> local_element_volume = element_volume[cell];

      // loop over quadrature points
      for (unsigned int q = 0; q < num_q_points; ++q)
        {
          variable_list.q_point    = q;
          pp_variable_list.q_point = q;

          dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc =
            variable_list.get_q_point_location();

          // Calculate the residuals
          postProcessedFields(variable_list,
                              pp_variable_list,
                              q_point_loc,
                              local_element_volume);
        }

      pp_variable_list.integrate_and_distribute(dst);
    }
}

#include "../../include/matrixFreePDE_template_instantiations.h"
