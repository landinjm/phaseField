#ifndef INCLUDE_INITIALCONDITIONS_H_
#define INCLUDE_INITIALCONDITIONS_H_

#include <core/matrixFreePDE.h>
#include <core/userInputParameters.h>
#include <field_input/read_vtk.h>

template <int dim, int degree>
class InitialCondition : public dealii::Function<dim>
{
public:
  const unsigned int             index;
  const userInputParameters<dim> userInputs;
  dealii::Vector<double>         values;

  InitialCondition(const unsigned int             _index,
                   const userInputParameters<dim> _userInputs,
                   MatrixFreePDE<dim, degree>    *_matrix_free_pde)
    : dealii::Function<dim>(1)
    , index(_index)
    , userInputs(_userInputs)
    , matrix_free_pde(_matrix_free_pde)
  {
    std::srand(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) + 1);
  }

  // IC for scalar values
  [[nodiscard]] double
  value(const dealii::Point<dim> &p, const unsigned int component = 0) const override
  {
    double                 scalar_IC = 0.0;
    dealii::Vector<double> vector_IC(dim);

    matrix_free_pde->setInitialCondition(p, index, scalar_IC, vector_IC);
    return scalar_IC;
  };

private:
  MatrixFreePDE<dim, degree> *matrix_free_pde;
};

template <int dim, int degree>
class InitialConditionVector : public dealii::Function<dim>
{
public:
  const unsigned int             index;
  const userInputParameters<dim> userInputs;
  dealii::Vector<double>         values;

  InitialConditionVector(const unsigned int             _index,
                         const userInputParameters<dim> _userInputs,
                         MatrixFreePDE<dim, degree>    *_matrix_free_pde)
    : dealii::Function<dim>(dim)
    , index(_index)
    , userInputs(_userInputs)
    , matrix_free_pde(_matrix_free_pde)
  {
    std::srand(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) + 1);
  }

  // IC for vector values
  void
  vector_value(const dealii::Point<dim> &p,
               dealii::Vector<double>   &vector_IC) const override
  {
    double scalar_IC = 0.0;
    vector_IC.reinit(dim);
    matrix_free_pde->setInitialCondition(p, index, scalar_IC, vector_IC);
  };

private:
  MatrixFreePDE<dim, degree> *matrix_free_pde;
};

template <int dim, typename datatype>
class ScalarInitialConditionVTK : public Function<dim>
{
public:
  ScalarInitialConditionVTK(
    const boost::unordered_map<dealii::Point<dim, datatype>,
                               datatype,
                               std::hash<dealii::Point<dim, datatype>>> &scalar_mappings)
    : dealii::Function<dim>(1)
    , scalar_mappings(scalar_mappings)
  {}

  void
  value_list(const std::vector<dealii::Point<dim, datatype>> &points,
             std::vector<datatype>                           &values,
             const unsigned int component = 0) const override
  {
    for (std::size_t i = 0; i < points.size(); ++i)
      {
        const auto &point = points[i];

        auto it = scalar_mappings.find(point);

        if (it != scalar_mappings.end())
          {
            values[i] = it->second;
          }
        else
          {
            values[i] = datatype {};
          }
      }
  };

private:
  const boost::unordered_map<dealii::Point<dim, datatype>,
                             datatype,
                             std::hash<dealii::Point<dim, datatype>>>
    scalar_mappings;
};

#endif
