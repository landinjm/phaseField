#include <prismspf/core/problem.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

TEMPLATE_TEST_CASE("Explicit Allen-Cahn Regression",
                   "[regression][explicit][allen_cahn]",
                   float,
                   double)
{
  using namespace prisms;

  using number = TestType;

  // TODO: Make these templates
  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 2;

  // TODO: How and where do I initialize MPI

  // Define PDE
  class CustomPDE : public PDEOperatorBase<dim, degree, number>
  {
  public:
    CustomPDE(const UserInputParameters<dim> &_user_inputs,
              PhaseFieldTools<dim>           &_pf_tools)
      : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools) {};

  private:
    void
    set_initial_condition([[maybe_unused]] const unsigned int       &index,
                          [[maybe_unused]] const unsigned int       &component,
                          [[maybe_unused]] const dealii::Point<dim> &point,
                          [[maybe_unused]] number                   &scalar_value,
                          [[maybe_unused]] number &vector_component_value) const override
    {}

    void
    compute_rhs(FieldContainer<dim, degree, number> &variable_list,
                const SimulationTimer               &sim_timer,
                unsigned int                         solve_block_id) const override
    {}
  };

  // Define the fields and solve blocks
  std::vector<FieldAttributes> fields = {FieldAttributes("phi")};

  SolveBlock phi_block;
  phi_block.id            = 0;
  phi_block.solve_type    = Explicit;
  phi_block.solve_timing  = Initialized;
  phi_block.field_indices = {0};
  phi_block.dependencies_rhs =
    make_dependency_set(fields, {"old_1(phi)", "grad(old_1(phi))"});

  std::vector<SolveBlock> solve_blocks({phi_block});

  // Define the parameters
  UserInputParameters<dim> user_inputs;

  // Other miscellaneous things
  PhaseFieldTools<dim>         pf_tools;
  CustomPDE                    pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, number> problem(fields,
                                       solve_blocks,
                                       user_inputs,
                                       pf_tools,
                                       pde_operator);

  // Init system and create the simulation timer.
  problem.init_system();
}
