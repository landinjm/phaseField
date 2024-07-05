
#include "../../include/matrixFreePDE.h"

template <int dim, int degree>
void  MatrixFreePDE<dim,degree>::computeIntegral(double& integratedField, int index, std::vector<vectorType*> variableSet) {
	//Check that input index is a scalar field
	if (fields[index].type != SCALAR) {
		std::cerr << "TypeError: double integratedField does not match the index field" << std::endl;
		abort();
	}

	//Grab the requisite parts of the field for integration
	QGauss<dim>  quadrature_formula(degree+1);
	FE_Q<dim> FE (QGaussLobatto<1>(degree+1));
	FEValues<dim> fe_values (FE, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
	const unsigned int   n_q_points    = quadrature_formula.size();
	std::vector<double> cVal(n_q_points);

	double value = 0.0;

	//Loop over the active cells and find the integrated value by taking the product of the quad values & quad weights
	for (auto cell=this->dofHandlersSet[index]->begin_active(); cell!=this->dofHandlersSet[index]->end(); ++cell) {
		if (cell->is_locally_owned()){
			fe_values.reinit (cell);

			fe_values.get_function_values(*variableSet[index], cVal);

			for (unsigned int q=0; q<n_q_points; ++q){
				value+=(cVal[q])*fe_values.JxW(q);
			}
		}
	}

	//Grab the sum over all processors
	value=Utilities::MPI::sum(value, MPI_COMM_WORLD);

	integratedField = value;
}

template <int dim, int degree>
void  MatrixFreePDE<dim,degree>::computeIntegral(std::vector<double>& integratedField, int index, std::vector<vectorType*> variableSet) {
	//Check that input index is a vector field
	if (fields[index].type != VECTOR) {
		std::cerr << "TypeError: vector<double> integratedField integratedField does not match the index field" << std::endl;
		abort();
	}

	//Resize the integratedField vector to match the number of dimensions
	integratedField.resize(dim, 0.0); 

	//Grab the requisite parts of the field for integration
	QGauss<dim>  quadrature_formula(degree+1);
	FESystem<dim> FE (FE_Q<dim>(QGaussLobatto<1>(degree+1)),dim);
	FEValues<dim> fe_values (FE, quadrature_formula, update_values | update_JxW_values | update_quadrature_points);
	const unsigned int   n_q_points    = quadrature_formula.size();
	std::vector<dealii::Vector<double>> cVal(n_q_points,dealii::Vector<double>(dim));

	double value[dim] = {0.0};

	//Loop over the active cells and find the integrated value by taking the product of the quad values & quad weights
	for (auto cell=this->dofHandlersSet[index]->begin_active(); cell!=this->dofHandlersSet[index]->end(); ++cell) {
		if (cell->is_locally_owned()){
			fe_values.reinit (cell);

			fe_values.get_function_values(*variableSet[index], cVal);

			for (unsigned int q=0; q<n_q_points; ++q){
				for (unsigned int i=0; i<dim; ++i){
					value[i] += (cVal[q][i])*fe_values.JxW(q);
				}
			}
		}
	}

	//Grab the sum over all processors
	for (unsigned int i=0; i<dim; ++i){
		value[i] = Utilities::MPI::sum(value[i], MPI_COMM_WORLD);

		integratedField[i] = value[i];
	}
}

#include "../../include/matrixFreePDE_template_instantiations.h"
