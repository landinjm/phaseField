// base class for matrix Free implementation of PDE's
#ifndef MATRIXFREEPDE_H
#define MATRIXFREEPDE_H

// general headers
#include <fstream>
#include <iterator>
#include <sstream>

// dealii headers
#include <deal.II/base/quadrature.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#if (DEAL_II_VERSION_MAJOR == 9 && DEAL_II_VERSION_MINOR > 3)
#include <deal.II/fe/mapping_fe.h>
#endif
#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/solver_control.h>

// PRISMS headers
#include "SimplifiedGrainRepresentation.h"
#include "adaptiveRefinement.h"
#include "discretization.h"
#include "boundaryConditions.h"
#include "fields.h"
#include "nucleus.h"
#include "userInputParameters.h"
#include "variableContainer.h"
#include "variableValueContainer.h"

// define data types
#ifndef scalarType
typedef dealii::VectorizedArray<double> scalarType;
#endif
#ifndef vectorType
typedef dealii::LinearAlgebra::distributed::Vector<double> vectorType;
#endif

// macro for constants
#define constV(a) make_vectorized_array(a)

//
using namespace dealii;
//
// base class for matrix free PDE's
//
/**
 * This is the abstract base class for the matrix free implementation of Parabolic and Elliptic BVP's,
 * and supports MPI, Threads and Vectorization (Hybrid Parallel).
 * This class contains the parallel data structures, mesh (referred to as triangulation),
 * parallel degrees of freedom distribution,  constraints,  and general utility methods.
 *
 * All the physical models in this package inherit this base class.
 */
template <int dim, int degree>
class MatrixFreePDE : public Subscriptor {
public:
    /**
     * Class contructor
     */
    MatrixFreePDE(userInputParameters<dim>);
    ~MatrixFreePDE();
    
    virtual void init();

    /**
     * Initializes the data structures for enabling unit tests.
     *
     * This method initializes the MatrixFreePDE object with a fixed geometry, discretization and
     * other custom selected options specifically to help with unit tests, and should not be called
     * in any of the physical models.
     */
    void initForTests(std::vector<Field<dim>> _fields);

    /**
     * This method implements the time stepping algorithm and invokes the solveIncrement() method.
     */
    void solve();
    /**
     * This method essentially converts the MatrixFreePDE object into a matrix object which can be
     * used with matrix free iterative solvers. Provides the A*x functionality for solving the system of
     * equations AX=b.
     */
    void vmult(vectorType& dst, const vectorType& src) const;
    /**
     * Vector of all the physical fields in the problem. Fields are identified by dimentionality (SCALAR/VECTOR),
     * the kind of PDE (ELLIPTIC/PARABOLIC) used to compute them and a character identifier  (e.g.: "c" for composition)
     * which is used to write the fields to the output files.
     */
    std::vector<Field<dim>> fields;

    void buildFields();

    // Parallel message stream
    ConditionalOStream pcout;

    // Initial conditions function
    virtual void setInitialCondition(const dealii::Point<dim>& p, const unsigned int index, double& scalar_IC, dealii::Vector<double>& vector_IC) = 0;

    virtual void setNonUniformDirichletBCs(const dealii::Point<dim>& p, const unsigned int index, const unsigned int direction, const double time, double& scalar_BC, dealii::Vector<double>& vector_BC) = 0;

protected:
    userInputParameters<dim> userInputs;

    /*Discretization*/
    discretization<dim> Discretization;

    /*Boundary Conditions*/
    boundaryConditions<dim, degree> BCs;

    /*AMR methods*/
    adaptiveRefinement<dim, degree> RefineAdaptively;

    // Virtual methods to set the attributes of the primary field variables and the postprocessing field variables
    // virtual void setVariableAttriubutes() = 0;
    // virtual void setPostProcessingVariableAttriubutes(){};
    variableAttributeLoader var_attributes;

    // Elasticity matrix variables
    const static unsigned int CIJ_tensor_size = 2 * dim - 1 + dim / 3;

    // Method to reinitialize the mesh, degrees of freedom, constraints and data structures when the mesh is adapted
    void reinit();

    /**
     * Method to reassign grains when multiple grains are stored in a single order parameter.
     */
    void reassignGrains();

    std::vector<SimplifiedGrainRepresentation<dim>> simplified_grain_representations;

    /**
     * Method to solve each time increment of a time-dependent problem. For time-independent problems
     * this method is called only once. This method solves for all the fields in a staggered manner (one after another)
     * and also invokes the corresponding solvers: Explicit solver for Parabolic problems, Implicit (matrix-free) solver for Elliptic problems.
     */
    virtual void solveIncrement(bool skip_time_dependent);
    /* Method to write solution fields to vtu and pvtu (parallel) files.
     *
     * This method can be enabled/disabled by setting the flag writeOutput to true/false. Also,
     * the user can select how often the solution files are written by setting the flag
     * skipOutputSteps in the parameters file.
     */
    void outputResults();

    /*Vector all the solution vectors in the problem. In a multi-field problem, each primal field has a solution vector associated with it.*/
    std::vector<vectorType*> solutionSet;
    /*Vector all the residual (RHS) vectors in the problem. In a multi-field problem, each primal field has a residual vector associated with it.*/
    std::vector<vectorType*> residualSet;
    /*Vector of parallel solution transfer objects. This is used only when adaptive meshing is enabled.*/
    std::vector<parallel::distributed::SolutionTransfer<dim, vectorType>*> soltransSet;

    /*Vector to store the inverse of the mass matrix diagonal for scalar and vector fields. Due to the choice of spectral elements with Guass-Lobatto quadrature, the mass matrix is diagonal.*/
    vectorType invMscalar, invMvector;

    /*Vector to store the solution increment. This is a temporary vector used during implicit solves of the Elliptic fields.*/
    vectorType dU_vector, dU_scalar;

    // matrix free methods
    /*Current field index*/
    unsigned int currentFieldIndex;
    /*Method to compute the inverse of the mass matrix*/
    void computeInvM();

    /*Method to compute an explicit timestep*/
    void updateExplicitSolution(unsigned int fieldIndex);

    void applyBCs(unsigned int fieldIndex);

    /*Method to print outputs*/
    void printOutputs(unsigned int fieldIndex, SolverControl *solver_control = nullptr);

    /*Method for nonlinear solve*/
    bool nonlinearSolve(unsigned int fieldIndex, unsigned int nonlinear_it_index);

    /*Method to compute the right hand side (RHS) residual vectors*/
    void computeExplicitRHS();
    void computeNonexplicitRHS();

    // virtual methods to be implemented in the derived class
    /*Method to calculate LHS(implicit solve)*/
    void getLHS(const MatrixFree<dim, double>& data,
        vectorType& dst,
        const vectorType& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const;

    bool generatingInitialGuess;
    void getLaplaceLHS(const MatrixFree<dim, double>& data,
        vectorType& dst,
        const vectorType& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const;

    void setNonlinearEqInitialGuess();
    void computeLaplaceRHS(unsigned int fieldIndex);
    void getLaplaceRHS(const MatrixFree<dim, double>& data,
        vectorType& dst,
        const vectorType& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*Method to calculate RHS (implicit/explicit). This is an abstract method, so every model which inherits MatrixFreePDE<dim> has to implement this method.*/
    void getExplicitRHS(const MatrixFree<dim, double>& data,
        std::vector<vectorType*>& dst,
        const std::vector<vectorType*>& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const;

    void getNonexplicitRHS(const MatrixFree<dim, double>& data,
        std::vector<vectorType*>& dst,
        const std::vector<vectorType*>& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const;

    virtual void explicitEquationRHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const = 0;

    virtual void nonExplicitEquationRHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const = 0;

    virtual void equationLHS(variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const = 0;

    virtual void postProcessedFields(const variableContainer<dim, degree, dealii::VectorizedArray<double>>& variable_list,
        variableContainer<dim, degree, dealii::VectorizedArray<double>>& pp_variable_list,
        const dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc) const {};
    void computePostProcessedFields(std::vector<vectorType*>& postProcessedSet);

    void getPostProcessedFields(const dealii::MatrixFree<dim, double>& data,
        std::vector<vectorType*>& dst,
        const std::vector<vectorType*>& src,
        const std::pair<unsigned int, unsigned int>& cell_range);

    
    std::vector<std::map<dealii::types::global_dof_index, double>*> valuesDirichletSet;
    
    void applyDirichletBCs();

    void getComponentsWithRigidBodyModes(std::vector<int>&) const;
    void setRigidBodyModeConstraints(const std::vector<int>, AffineConstraints<double>*, const DoFHandler<dim>*) const;

    // methods to apply initial conditions
    /*Virtual method to apply initial conditions.  This is usually expected to be provided by the user in IBVP (Initial Boundary Value Problems).*/

    void applyInitialConditions();

    // --------------------------------------------------------------------------
    // Methods for saving and loading checkpoints
    // --------------------------------------------------------------------------

    void save_checkpoint();

    void load_checkpoint_triangulation();
    void load_checkpoint_fields();
    void load_checkpoint_time_info();

    void move_file(const std::string&, const std::string&);

    void verify_checkpoint_file_exists(const std::string filename);

    // --------------------------------------------------------------------------
    // Nucleation methods and variables
    // --------------------------------------------------------------------------
    // Vector of all the nuclei seeded in the problem
    std::vector<nucleus<dim>> nuclei;

    // Method to get a list of new nuclei to be seeded
    void updateNucleiList();
    std::vector<nucleus<dim>> getNewNuclei();
    void getLocalNucleiList(std::vector<nucleus<dim>>& newnuclei) const;
    void safetyCheckNewNuclei(std::vector<nucleus<dim>> newnuclei, std::vector<unsigned int>& conflict_ids);
    void refineMeshNearNuclei(std::vector<nucleus<dim>> newnuclei);
    double weightedDistanceFromNucleusCenter(const dealii::Point<dim, double> center, const std::vector<double> semiaxes, const dealii::Point<dim, double> q_point_loc, const unsigned int var_index) const;
    dealii::VectorizedArray<double> weightedDistanceFromNucleusCenter(const dealii::Point<dim, double> center, const std::vector<double> semiaxes, const dealii::Point<dim, dealii::VectorizedArray<double>> q_point_loc, const unsigned int var_index) const;

    // Method to obtain the nucleation probability for an element, nontrival case must be implemented in the subsclass
    virtual double getNucleationProbability(variableValueContainer, double, dealii::Point<dim>, unsigned int variable_index) const { return 0.0; };

    // utility functions
    /*Returns index of given field name if exists, else throw error.*/
    unsigned int getFieldIndex(std::string _name);

    std::vector<double> freeEnergyValues;
    void outputFreeEnergy(const std::vector<double>& freeEnergyValues) const;

    /*Method to compute the integral of a field.*/
    void computeIntegral(double& integratedField, int index, std::vector<vectorType*> postProcessedSet);

    // variables for time dependent problems
    /*Flag used to see if invM, time stepping in run(), etc are necessary*/
    bool isTimeDependentBVP;
    /*Flag used to mark problems with Elliptic fields.*/
    bool isEllipticBVP;

    bool hasExplicitEquation;
    bool hasNonExplicitEquation;
    //
    double currentTime;
    unsigned int currentIncrement, currentOutput, currentCheckpoint, current_grain_reassignment;

    /*Timer and logging object*/
    mutable TimerOutput computing_timer;

    std::vector<double> integrated_postprocessed_fields;

    bool first_integrated_var_output_complete;

    // Methods and variables for integration
    double integrated_var;
    unsigned int integral_index;
    std::mutex assembler_lock;

    void computeIntegralMF(double& integratedField, int index, const std::vector<vectorType*> postProcessedSet);

    void getIntegralMF(const MatrixFree<dim, double>& data,
        std::vector<vectorType*>& dst,
        const std::vector<vectorType*>& src,
        const std::pair<unsigned int, unsigned int>& cell_range);
};

#endif
