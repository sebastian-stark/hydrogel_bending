#include <iostream>
#include <fstream>

#include <boost/math/special_functions/lambert_w.hpp>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <incremental_fe/fe_model.h>
#include <incremental_fe/scalar_functionals/psi_lib.h>
#include <incremental_fe/scalar_functionals/omega_lib.h>

using namespace std;
using boost::math::lambert_w0;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

// function implementing time stepping based on geometric series, with smaller time increments in the beginning of the time interval under consideration if needed (put q=1 for equidistant time increments)
// t_0 : begin of time interval
// t_1 : end of time interval
// N   : number of time increments covering the time interval
// q   : ratio between two subsequent time increments
// m   : 2^m is the number of equally spaced sub-time-increments to be used for each time increment
// returned are the time instants t_k, where k = 0 ... 2^m * N
vector<double> get_time_instants(	const double 		t_0,
									const double 		t_1,
									const unsigned int	N,
									const double		q,
									const unsigned int	m)
{
	vector<double> time_instants;

	vector<double> time_increments_unrefined;
	for(unsigned int k = 0; k < N; ++k)
	{
		if(q != 1.0)
			time_increments_unrefined.push_back( (t_1 - t_0) * pow(q, (double)(N - 1 - k)) / ( ( 1.0 - pow(q, (double)N) ) / ( 1.0 - q ) ) );
		else
			time_increments_unrefined.push_back( (t_1 - t_0) / (double)N );
	}

	time_instants.push_back(t_0);
	const unsigned int I = pow(2.0, (double)m ) + 0.5;
	for(const auto& time_increment : time_increments_unrefined)
	{
		for(unsigned int i = 0; i < I; ++i)
			time_instants.push_back(time_instants.back() + time_increment / I);
	}

	return time_instants;
}

// Class implementing the trial step required for the simulation.
// This does, before each assembly of the finite element system, check whether the current value of phi_ref is such that the electrolysis is triggered somewhere or not.
// If the electrolysis is triggered, nothing is done. Otherwise, the finite element system
// matrix is regularized by setting the diagonal element corresponding to phi_ref to a small negative value in order to remove the non-uniqueness of phi_ref (and remove, therefore, the singularity of the finite element
// system matrix)
template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
class TrialStep
{
private:

	// the finite element model
	FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>&
	fe_model;

	// the scalar functionals describing the electrolysis reactions
	const set<const OmegaElectrolysis00<spacedim>*>
	electrolysis_scalar_functionals;

	// reference to the solution vector of the fe model
	SolutionVectorType&
	solution;

	// reference to the reference solution vector of the fe model
	SolutionVectorType&
	solution_ref;

	// reference to the system matrix of the fe model
	MatrixType&
	system_matrix;

	// the global dof index associated with the reference potential phi_ref
	const unsigned int
	dof_index_phi;

	// bool indicating whether the electrolysis process has been found to be active during the last trial step
	mutable bool
	electrolysis_active = false;

	// this is called before each assembly of the finite element system;
	// if the current value of phi_ref is such that no electrolysis would take place,
	// it is shifted to the upper bound not triggering an electrolysis current for the current solution
	void pre_assembly()
	const
	{
		// determine whether electrolysis is currently triggered
		for(auto& omega : electrolysis_scalar_functionals)
			omega->electrolysis_active = false;
		set<const ScalarFunctional<spacedim-1, spacedim>*> scalar_functionals;
		for(const auto& sf : electrolysis_scalar_functionals)
			scalar_functionals.insert(sf);
		fe_model.get_assembly_helper().call_scalar_functionals(solution, {&solution_ref}, set<const ScalarFunctional<spacedim, spacedim>*>(), scalar_functionals);
		electrolysis_active = false;
		for(auto& omega : electrolysis_scalar_functionals)
		{
			if(omega->electrolysis_active)
				electrolysis_active = true;
		}
	}

	// if the electrolysis is not active, regularize the system matrix after it has been assembled
	void post_assembly()
	const
	{
		if(!electrolysis_active)
		{
			system_matrix.add(dof_index_phi, dof_index_phi, -1e-12);
			system_matrix.compress(VectorOperation::add);
		}
	}

public:

	// constructor
	TrialStep(	FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>&	fe_model,
				const set<const OmegaElectrolysis00<spacedim>*>&					electrolysis_scalar_functionals,
				const IndependentField<0, spacedim>&								phi_ref)
	:
	fe_model(fe_model),
	electrolysis_scalar_functionals(electrolysis_scalar_functionals),
	solution(fe_model.get_solution_vector()),
	solution_ref(fe_model.get_solution_ref_vector()),
	system_matrix(fe_model.get_system_matrix()),
	dof_index_phi(fe_model.get_assembly_helper().get_global_dof_index_C(&phi_ref))
	{
		// connect  to the pre_assembly and post_assembly signals of the finite element model
		fe_model.pre_assembly.connect(boost::bind(&TrialStep<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::pre_assembly, this));
		fe_model.post_assembly.connect(boost::bind(&TrialStep<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::post_assembly, this));
	}

};

// class for computation of initial equilibrium state;
// currently works for monovalent salt only
class InitialState
{

private:

	// material constant (molar volume of water)
	const double
	V_m_H2O;

	// gas constant times temperature
	const double
	RT;

	// material constant (combination of some other constants)
	const double
	A;

	// initial concentrations in solution bath (A+, B-, H+, OH-)
	const Vector<double>
	c_i_s_ref;

	// signed charge of ionic species in multiples of unit charge
	const vector<int>
	z_i = {1, -1, 1, -1};

	// signed charge multiplicity of charged groups of gel in multiples of unit charge
	const int
	z_p = -1;

	// material constant (maximum possible charge density in dry state if hypothetically fully dissociated)
	const double
	c_p_max_0;

	// material constant (shear modulus)
	const double
	mu_g;

	// material constant (water affinity of gel)
	const double
	chi;

	// whether to take CH dissociation into account (if false, the CH groups are assumed to be fully dissociated)
	const bool
	with_CH_dissociation = true;

	// compute c_p and derivatives in dependence on initial swelling J_0 and potential difference phi (only if CH dissociation is accounted for)
	void
	get_c_p_and_derivatives(const double	J_0,
							const double	phi,
							double& 		c_p,
							double&			dc_p_dJ,
							double&			dc_p_dphi)
	const
	{
		tuple<double, double, double> c_p_and_derivatives;
		c_p = 0.0;
		dc_p_dJ = 0.0;
		dc_p_dphi = 0.0;
		for(unsigned int i = 0; i < c_i_s_ref.size(); ++i)
		{
			c_p += (1.0 - J_0) / J_0 * z_i[i]/z_p * c_i_s_ref[i] * exp(z_i[i] * phi);
			dc_p_dJ += -1.0 / J_0 / J_0 *  z_i[i]/z_p * c_i_s_ref[i] * exp(z_i[i] * phi);
			dc_p_dphi += (1.0 - J_0) / J_0 * z_i[i] * z_i[i] / z_p * c_i_s_ref[i] * exp(z_i[i] * phi);
		}
	}

	// compute c_i_g_ref and derivatives in dependence on initial swelling J_0 and potential difference phi
	void
	get_c_i_g_and_derivatives(	const double		J_0,
								const double		phi,
								Vector<double>&		c_i_g_ref,
								Vector<double>&		dc_i_g_dJ,
								Vector<double>&		dc_i_g_dphi)
	const
	{
		c_i_g_ref.reinit(c_i_s_ref.size());
		dc_i_g_dJ.reinit(c_i_s_ref.size());
		dc_i_g_dphi.reinit(c_i_s_ref.size());
		for(unsigned int i = 0; i < c_i_s_ref.size(); ++i)
		{
			c_i_g_ref[i] = (J_0 - 1.0) / J_0 * c_i_s_ref[i] * exp(z_i[i] * phi);
			dc_i_g_dJ[i] = 1.0 / J_0 / J_0 * c_i_s_ref[i] * exp(z_i[i] * phi);
			dc_i_g_dphi[i] = (J_0 - 1.0) / J_0 * c_i_s_ref[i] * z_i[i] * exp(z_i[i] * phi);
		}
	}

	// defines the nonlinear system to be satisfied by initial swelling J_0 and potential difference phi
	void
	get_nonlinear_system(	const double 		J_0,
							const double 		phi,
							Vector<double>& 	f,
							FullMatrix<double>&	H)
	const
	{
		f.reinit(2);
		H.reinit(2, 2);

		Vector<double> c_i_g, dc_i_g_dJ, dc_i_g_dphi;
		get_c_i_g_and_derivatives(J_0, phi, c_i_g, dc_i_g_dJ, dc_i_g_dphi);
		double sum_c_i_s = 0;
		double sum_c_i_g = 0;
		double sum_z_i_c_i_g = 0;
		double sum_dc_i_g_dJ = 0;
		double sum_dz_i_c_i_g_dJ = 0;
		double sum_dc_i_g_dphi = 0;
		double sum_dz_i_c_i_g_dphi = 0;
		for(const auto& val : c_i_s_ref)
			sum_c_i_s += val;
		for(const auto& val : c_i_g)
			sum_c_i_g += val;
		for(unsigned int i = 0; i < c_i_g.size(); ++i)
			sum_z_i_c_i_g += c_i_g[i] * z_i[i];
		for(const auto& val : dc_i_g_dJ)
			sum_dc_i_g_dJ += val;
		for(unsigned int i = 0; i < c_i_g.size(); ++i)
			sum_dz_i_c_i_g_dJ += dc_i_g_dJ[i] * z_i[i];
		for(const auto& val : dc_i_g_dphi)
			sum_dc_i_g_dphi += val;
		for(unsigned int i = 0; i < c_i_g.size(); ++i)
			sum_dz_i_c_i_g_dphi += dc_i_g_dphi[i] * z_i[i];

		if(with_CH_dissociation)
		{
			double c_p, dc_p_dJ, dc_p_dphi;
			get_c_p_and_derivatives(J_0, phi, c_p, dc_p_dJ, dc_p_dphi);
			f[0]    = - (A * J_0 * c_i_g[2] * c_p - (J_0 - 1.0) * (c_p_max_0/J_0 - c_p));
			H(0,0) = A * ( c_i_g[2] * c_p + J_0 *  dc_i_g_dJ[2] * c_p + J_0 * c_i_g[2] * dc_p_dJ ) - (c_p_max_0/J_0 - c_p) + (J_0 - 1.0) * ( dc_p_dJ + c_p_max_0 / J_0 / J_0 );
			H(0,1) = A * J_0 * (dc_i_g_dphi[2] * c_p + J_0 * c_i_g[2] * dc_p_dphi) + (J_0 - 1.0) * dc_p_dphi;
		}
		else
		{
			f[0] = sum_z_i_c_i_g + z_p * c_p_max_0 / J_0;
			H(0,0) = -sum_dz_i_c_i_g_dJ + z_p * c_p_max_0 / J_0 / J_0;
			H(0,1) = -sum_dz_i_c_i_g_dphi;
		}
		f[1]    = - ( mu_g / RT * ( pow(J_0, -1.0/3.0) - 1.0 / J_0 ) + 1.0 / V_m_H2O * ( log(1.0 - 1.0 / J_0) + 1.0 / J_0 + chi / J_0 / J_0 ) + sum_c_i_s - J_0 / (J_0 - 1.0) * sum_c_i_g );
		H(1,0) = mu_g / RT * ( -1.0/3.0 * pow(J_0, -4.0/3.0) + 1.0 / J_0 /J_0 ) + 1.0 / V_m_H2O * ( 1.0 / ( J_0 * J_0 - J_0 ) - 1.0 / J_0 / J_0 - 2.0 * chi / J_0 / J_0 / J_0 ) + 1.0 / (J_0 - 1.0) / (J_0 - 1.0) * sum_c_i_g - J_0 / (J_0 - 1.0) * sum_dc_i_g_dJ;
		H(1,1) = - J_0 / (J_0 - 1.0) * sum_dc_i_g_dphi;
	}

public:

	// constructor
	InitialState(	const double mu_0_H,
					const double delta_mu_p,
					const double V_m_H2O,
					const double RT,
					const double c_A_B_ref,
					const double c_H_OH_ref,
					const double c_p_max_0,
					const double mu_p,
					const double chi,
					const bool with_CH_dissociation = true)
	:
	V_m_H2O(V_m_H2O),
	RT(RT),
	A( V_m_H2O * exp( (mu_0_H + delta_mu_p) / RT ) ),
	c_i_s_ref({c_A_B_ref, c_A_B_ref, c_H_OH_ref, c_H_OH_ref}),
	c_p_max_0(c_p_max_0),
	mu_g(mu_p),
	chi(chi),
	with_CH_dissociation(with_CH_dissociation)
	{}

	// return the initial state (ion concentrations c_i_g_ref, initial polymer volume fraction n_p_ref, referential molar concentration of charged groups of polymer, initial pressure in gel)
	void
	get_initial_state(	Vector<double>& c_i_g_ref,
						double&			n_p_ref,
						double&			c_p_ref,
						double&			p_g_ref)
	{
		Vector<double> x(2), dx(2);
		x[0] = 100.0;
		x[1] = 1.0;
		Vector<double> f(2);
		FullMatrix<double> H(2,2), H_inv(2,2);
		for(;;)
		{
			get_nonlinear_system(x[0], x[1], f, H);
			H_inv.invert(H);
			H_inv.vmult(dx, f);
			for(unsigned int i = 0; i < x.size(); ++i)
				x[i] += dx[i];
			if(dx.l2_norm() < 1e-10)
				break;
		}
		const double J = x[0];
		const double phi = x[1];
		double dc_p_dJ, dc_p_dphi;
		Vector<double> dc_i_g_dJ, dc_i_g_dphi;
		if(with_CH_dissociation)
			get_c_p_and_derivatives(J, phi, c_p_ref, dc_p_dJ, dc_p_dphi);
		else
			c_p_ref = c_p_max_0 / J;
		get_c_i_g_and_derivatives(J, phi, c_i_g_ref, dc_i_g_dJ, dc_i_g_dphi);
		n_p_ref = 1.0 / J;
		p_g_ref = 	mu_g * n_p_ref * (pow(n_p_ref, -2.0/3.0) - 1.0);
	}

};

// post-processor computing concentration of a species referred to actual volume
template<unsigned int spacedim>
class PostprocessorConcentration : public DataPostprocessorScalar<spacedim>
{
	// global component index of the referential concentration variable
	const unsigned int
	global_component_index_c;

	// global component index of first component of displacement field
	const unsigned int
	global_component_index_u;

public:

	// constructor
	PostprocessorConcentration(	const string&		name,
								const unsigned int	global_component_index_c,
								const unsigned int	global_component_index_u)
	:
	DataPostprocessorScalar<spacedim>(name, update_gradients),
	global_component_index_c(global_component_index_c),
	global_component_index_u(global_component_index_u)
	{
	}

	void
	evaluate_vector_field(	const DataPostprocessorInputs::Vector<spacedim>&	input_data,
							vector<Vector<double>>&								computed_quantities)
	const
	{
		for(unsigned int dataset = 0; dataset < input_data.solution_gradients.size(); ++dataset)
		{
			Tensor<2, spacedim> F;
			for(unsigned int m = 0; m < spacedim; ++m)
			{
				F[m][m] += 1.0;
				for(unsigned int n = 0; n < spacedim; ++n)
					F[m][n] += input_data.solution_gradients[dataset][global_component_index_u + m][n];
			}
			const double J = determinant(F);
			computed_quantities[dataset][0] = input_data.solution_values[dataset][global_component_index_c] / J;
		}
	}
};

// post-processor computing pH value
template<unsigned int spacedim>
class PostprocessorPH : public DataPostprocessorScalar<spacedim>
{
	// global component index of the referential concentration variable
	const unsigned int
	global_component_index_c;

	// global component index of first component of displacement field
	const unsigned int
	global_component_index_u;

	// global component index of water concentration
	const unsigned int
	global_component_index_c_H2O;

	// reference concentration of H+ in gel
	const double
	c_H_g_ref;

	// reference concentration of H+ in solution bath
	const double
	c_H_s_ref;

	// reference water concentration in gel
	const double
	c_H2O_g_ref;

	// reference water concentration in solution bath
	const double
	c_H2O_s_ref;

	// initial volume fraction of polymeric backbone
	const double
	n_p_ref;

	// standard concentration
	const double
	c_std;

	// regularization threshold (this is needed to compensate for the regularization approach and get the correct pH values)
	const double
	eps;

public:

	// constructor
	PostprocessorPH(	const string&		name,
						const unsigned int	global_component_index_c,
						const unsigned int	global_component_index_u,
						const unsigned int	global_component_index_c_H2O,
						const double		c_H_g_ref,
						const double		c_H_s_ref,
						const double		c_H2O_g_ref,
						const double		c_H2O_s_ref,
						const double		n_p_ref,
						const double 		c_std,
						const double		eps = 0.0)
	:
	DataPostprocessorScalar<spacedim>(name, update_gradients),
	global_component_index_c(global_component_index_c),
	global_component_index_u(global_component_index_u),
	global_component_index_c_H2O(global_component_index_c_H2O),
	c_H_g_ref(c_H_g_ref),
	c_H_s_ref(c_H_s_ref),
	c_H2O_g_ref(c_H2O_g_ref),
	c_H2O_s_ref(c_H2O_s_ref),
	n_p_ref(n_p_ref),
	c_std(c_std),
	eps(eps)
	{
	}

	void
	evaluate_vector_field(	const DataPostprocessorInputs::Vector<spacedim>&	input_data,
							vector<Vector<double>>&								computed_quantities)
	const
	{

		const typename DoFHandler<spacedim>::cell_iterator current_cell = input_data.template get_cell<spacedim>();
		const unsigned int mat_id = current_cell->material_id();

		const unsigned int N = input_data.solution_gradients.size();

		for(unsigned int dataset = 0; dataset < N; ++dataset)
		{
			Tensor<2, spacedim> F;
			for(unsigned int m = 0; m < spacedim; ++m)
			{
				F[m][m] += 1.0;
				for(unsigned int n = 0; n < spacedim; ++n)
					F[m][n] += input_data.solution_gradients[dataset][global_component_index_u + m][n];
			}
			const double J = determinant(F);
			const double c = input_data.solution_values[dataset][global_component_index_c];
			const double c_H2O = input_data.solution_values[dataset][global_component_index_c_H2O];
			const double c_H_ref = mat_id == 0 ? c_H_g_ref : c_H_s_ref;
			const double c_H2O_ref = mat_id == 0 ? c_H2O_g_ref : c_H2O_s_ref;
			const double n_ref = mat_id == 0 ? n_p_ref : 0.0;
			// compensating for regularization approach (this is done by using that the chemical potential of H+ ions is accurately represented by the regularized formulation, so that the correct value for
			// the H+ concentration can be calculated a posteriori)
			const double c_compensated = (c / c_H2O) / (c_H_ref / c_H2O_ref) > eps ? c : c_H2O * exp(log(c_H_ref / c_H2O_ref) + c_H2O / c * c_H_ref / c_H2O_ref * eps * log(eps));
			const double pH = c_compensated > 0.0 ? -log10(( c_compensated / (J - n_ref)) / c_std ) : 1e16;
			computed_quantities[dataset][0] = pH;
		}
	}
};

// class for piecewise constant initial state
template<unsigned int dim, unsigned int spacedim>
class ConstantFunctionPiecewise : public FunctionCell<dim, spacedim>
{
private:

	// value to be assigned for cells with material id 0
	double
	val_0;

	// value to be assigned for cells with material id other than 0
	double
	val_1;

public:

	// constructor
	ConstantFunctionPiecewise(	const double val_0,
								const double val_1)
	:
	FunctionCell<dim, spacedim>(),
	val_0(val_0),
	val_1(val_1)
	{
	}

	double
	value(	const Point<spacedim>&	/*p*/,
			const unsigned int		/*component=0*/)
	const
	{
		if(this->get_cell()->material_id() == 0)
			return val_0;
		else
			return val_1;
	}
};

// Function for prescribing external electrical potential on electrodes
template <unsigned int spacedim>
class UExternal : public Function<spacedim>
{

private:

	// constant part
	const double
	U_th;

	// change
	const double
	delta_U;

	// time t_1
	const double
	t_1;

public:

	// constructor
	UExternal(	const double U_th,
				const double delta_U,
				const double t_1)
	:
	Function<spacedim>(),
	U_th(U_th),
	delta_U(delta_U),
	t_1(t_1)
	{
	}

	double
	value(	const Point<spacedim>&	p,
			const unsigned int		/*component=0*/)
	const
	{
		const double t = this->get_time();
		// potential zero on cathode and non-zero on anode (anode at positive x, cathode at negative x)
		if(p(0) > 0.0)
		{
			if(t < t_1)
				return U_th + delta_U * t / t_1;
			else
				return U_th + delta_U;
		}
		else
			return 0.0;
	}
};

// Function defining scaling of Lame parameters within solution bath domain dependent on spatial position (done in order to reduce mesh distortion)
template <unsigned int spacedim>
class LameScaling : public Function<spacedim>
{
private:

	// half width of solid domain
	const double
	B_g_05;

	// scaling ratio (if > 1.0, the middle part of the solution bath domain is stiffer)
	const double
	ratio;

public:

	// constructor
	LameScaling(const double B_g_05,
				const double ratio)
	:
	Function<spacedim>(),
	B_g_05(B_g_05),
	ratio(ratio)
	{
	}

	double
	value(	const Point<spacedim>&	p,
			const unsigned int		/*component=0*/)
	const
	{
		const double x = fabs(p(0));
		double val = 1.0;
		if(x < 2.0 * B_g_05)
			val = ratio;
		else
			val = 1.0;
		return val;
	}
};

// the main function
vector<double>									// time increment size dt, individual errors in infinity norm, individual errors in l2 norm, combined error in infinity norm, combined error in l2 norm (errors only computed if write_reference==false)
solve(	const unsigned int 	m_t,				// number of refinements in time
		const unsigned int 	m_h,				// number of refinements in space
		const double		alpha,				// time integration parameter alpha
		const unsigned int	method,				// time integration method (0 - variationally consistent, 1 - alpha family, 2 - modified alpha family)
		const unsigned int	degree,				// degree of polynomial approximation of finite elements (1 - linear, 2 - quadratic, etc.)
		const string 		result_file,		// if write_reference == true: file into which solution vector is stored, if write_reference == false: file containing solution vector to compare with
		const bool			write_reference,	// whether to write reference solution or to compare with it
		const unsigned int	m_h_reference,		// number of refinements in space of reference solution
		const bool			write_output 		// whether to write output for each time increment to files (may cause a large number files), if false, only the results of the last time increment are written
		)
{
/********************
 * parameters, etc. *
 ********************/

	const unsigned int spacedim = 2;											// spatial dimensionality; this does only work for spacedim == 2
	const bool with_CH_dissociation = false;									// whether to consider the dissociation of the polymer
	const bool with_infinitesimal_viscosity = false;							// whether to account for infinitesimal viscosity and corresponding no-slip condition (if false, potential flow assumption is used)

	// quantities used for normalization (in SI units)
	const double c_ast = 100.0;													// mol/m^3
	const double R_ast = 8.31446261815324;										// J/mol/K
	const double T_ast = 293.15;												// K
	const double L_ast = 0.0125;												// m
	const double D_ast = 1.0e-8;												// m^2/s
	const double F_ast = 96485.33;												// C/mol

	// parameters
	const double B_g = 0.0025/L_ast;											// width of hydrogel strip
	const unsigned int N_H = 4;													// ratio between height (measured without half spherical tip)) and width of hydrogel strip
	const unsigned int N_B = 5;													// width of the solution bath on either side of the hydrogel strip in multiples of the width of the hydrogel strip

	const double R = 8.31446261815324 / R_ast;									// gas constant R = R_norm
	const double F = 96485.33 / F_ast;											// Faraday's constant F = F_norm
	const double T = 293.15 / T_ast;											// temperature T = T_norm

	const double c_A_s_ref = 50.0 / c_ast;										// initial concentration of A+ ions in external solution (mol/m^3)

	const double V_m_H2O = 1.80685e-05 * c_ast;									// molar volume of water (m^3/mol)
	const double c_std = 1000.0 / c_ast;										// standard concentration (1000 mol/m^3)
	const double mu_offset = R * T * log(c_std * V_m_H2O);						// offset for standard chemical potentials (used to compensate for different definition of chemical potentials than the usual one)
	const double mu_p = 300000.0 / (R_ast * T_ast * c_ast);						// Lame parameter of polymeric backbone (N/m^2)
	const double mu_s = mu_p;													// Lame parameter fictitious solid
	const double lambda_p = 0.0 / (R_ast * T_ast * c_ast);						// Lame parameter of polymeric backbone (N/m^2)
	const double lambda_s = lambda_p;											// Lame parameter fictitious solid (later scaled by scaling function to give a spatially dependent Lame parameter)
	const double psi_c_0 = -237130.0 / (R_ast * T_ast);							// free energy of formation of water (J/mol)
	const double mu_0_A = -261475.0 / (R_ast * T_ast) - mu_offset;				// chemical reference potential of A+ salt ions (J/mol)
	const double mu_0_B = -131260.0 / (R_ast * T_ast) - mu_offset;				// chemical reference potential of B- salt ions (J/mol)
	const double mu_0_H = -mu_offset;											// chemical reference potential of H+ (J/mol)
	const double mu_0_OH = -157290.0 / (R_ast * T_ast) - mu_offset;				// chemical reference potential of OH- (J/mol)
	const double delta_mu_CH = 25000.0 / (R_ast * T_ast);						// enthalpy change associated with dissociation of fixed anionic groups (J/mol)
	const double chi = 0.25;													// water affinity constant of gel
	const double z_A = 1.0;														// charge of A+ ions in multiples of absolute value of electron charge
	const double z_B = -1.0;													// charge of B- ions in multiples of absolute value of electron charge
	const double z_H = 1.0;														// charge of H+ in multiples of absolute value of electron charge
	const double z_OH = -1.0;													// charge of OH- in multiples of absolute value of electron charge
	const double z_C = -1.0;													// charge of ions bound to polymeric backbone in multiples of absolute value of electron charge
	const double D_H2O =  1e-10 * 8.31446261815324 * 293.15 / D_ast;			// Dissipation constant water (m^2/s)
	const double D_A =  5.4e-13 * 8.31446261815324 * 293.15 / D_ast;			// Dissipation constant A+ ions (m^2/s)
	const double D_B =  8.2e-13 * 8.31446261815324 * 293.15 / D_ast;			// Dissipation constant B- ions (m^2/s)
	const double D_H =  3.8e-12 * 8.31446261815324 * 293.15 / D_ast;			// Dissipation constant H+ ions (m^2/s)
	const double D_OH = 2.1e-12 * 8.31446261815324 * 293.15 / D_ast;			// Dissipation constant OH- ions (m^2/s)
	const double A_e_ox = -2.0;													// number of electrons flowing into solution for oxidation part of water electrolysis
	const double A_e_red = 1.0;													// number of electrons flowing into solution for reduction part of water electrolysis
	const double R_if_ox = 1.25e-5 * A_e_ox * A_e_ox * F_ast * F_ast * D_ast * c_ast
							/ L_ast / R_ast / T_ast;							// effective interface resistance for oxidation part of water electrolysis (Ohm m^2)
	const double R_if_red = 5e-5 * A_e_red * A_e_red * F_ast * F_ast * D_ast * c_ast
							/ L_ast / R_ast / T_ast;							// effective interface resistance for reduction part of water electrolysis (Ohm m^2)
	const double phi_c_ox = 0.125 * F_ast / (R_ast * T_ast);					// overpotential required to start oxidation part of water electrolysis (V)
	const double phi_c_red = -0.125 * F_ast / (R_ast * T_ast);					// overpotential required to start reduction part of water electrolysis (V)
	const double f = 0.5;														// fraction of charged groups of polymeric backbone
	const double V_p = (52.6e-6 + 17.75e-6 * f) * c_ast;						// molar volume of polymeric backbone in dry state (m^3/mol)

	// loading
	const double t = 600.0 * D_ast / L_ast / L_ast;								// time of computation
	const double delta_U = 2.25 * F_ast / (R_ast * T_ast);						// applied potential difference

	// numerical parameters
	const double eps = 1e-4;													// regularization parameter for chemical potentials at low concentrations
	const unsigned int N_refinements_global = m_h;								// number of global mesh refinements
	const unsigned int N_refinements_local_gs = 2;								// number of local mesh refinements at interface between gel and solution bath
	const unsigned int N_refinements_local_el = 2;								// number of local mesh refinements at electrodes
	const unsigned int cell_divisions = degree;									// number of element subdivisions for output
	const double q = 1.0;														// ratio between two subsequent time increments
	const unsigned int N = 32;													// initial number of time increments
	const bool use_local_elimination = true;									// whether to eliminate local dofs locally

	// mappings
	MappingQGeneric<spacedim, spacedim> mapping_domain(degree);					// FE mapping on domain
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(degree);			// FE mapping on interfaces

	// global data object, used to transfer global data (like time step information) between different potential contributions and to define parameters for the Newton-Raphson algorithm, etc.
	GlobalDataIncrementalFE<spacedim> global_data;

	// define some parameters for the problem solution
	global_data.set_compute_sparsity_pattern(1);								// compute sparsity pattern only once and re-use for subsequent steps
	global_data.set_max_iter(40);												// maximum number of Newton-Raphson iterations allowed
	global_data.set_threshold_residual(5e-14);									// convergence threshold for residual
	global_data.set_threshold_potential_increment(1e10);						// deactivate convergence check for potential increment
	global_data.set_perform_line_search(false);									// do not perform line search
	global_data.set_scale_residual(true);										// scale the residual according to the matrix diagonals

	// compute some other initial quantities such that initial state is an equilibrium state
	const double c_B_s_ref = c_A_s_ref;											// initial concentration of B- ions in external solution
	const double c_H_s_ref = lambert_w0( exp( (psi_c_0  - mu_0_H - mu_0_OH)		// initial concentration of H+ ions in external solution (calculated from dissociation equilibrium)
								/ (2.0 * R * T) - V_m_H2O * c_A_s_ref ) ) / V_m_H2O;
	const double c_OH_s_ref = c_H_s_ref;										// initial concentration of OH- ions in external solution
	InitialState initial_state(mu_0_H, delta_mu_CH, V_m_H2O, R*T, c_A_s_ref, c_H_s_ref, f / V_p, mu_p, chi, with_CH_dissociation);
	Vector<double> c_i_g_ref;
	double n_p_ref_, c_C_ref_, p_g_ref_;
	initial_state.get_initial_state(c_i_g_ref, n_p_ref_, c_C_ref_, p_g_ref_);
	const double c_A_g_ref  = c_i_g_ref[0];											// initial concentration of A+ ions in gel
	const double c_B_g_ref = c_i_g_ref[1];											// initial concentration of B- ions in gel
	const double c_H_g_ref = c_i_g_ref[2];											// initial concentration of OH- ions in gel
	const double c_OH_g_ref = c_i_g_ref[3];											// initial concentration of H+ ions in gel
	const double n_p_ref = n_p_ref_;												// initial volume fraction of polymer in gel
	const double c_C_ref = c_C_ref_;												// initial concentration of dissociated fixed charge groups in gel
	const double c_CH_ref = f / V_p * n_p_ref- c_C_ref;								// initial concentration of non-dissociated fixed charge groups in gel
	const double c_H2O_g_ref = (1.0 - n_p_ref) / V_m_H2O;							// initial fluid concentration in gel
	const double c_H2O_s_ref = 1.0 / V_m_H2O;										// initial fluid concentration in solution bath
	const double p_s_ref = 0.0;														// initial pressure in solution bath
	const double p_g_ref = p_g_ref_;												// initial pressure in gel
	const double eta_A_ref = mu_0_A + R * T * log(c_A_s_ref / c_H2O_s_ref)
	                       + mu_0_B  + R * T * log(c_B_s_ref / c_H2O_s_ref);		// initial equilibrium value for eta_A (not strictly required to be known, but it is easier to solve the first time increment starting with equilibrium value)
	const double eta_B_ref = 0.0;													// initial equilibrium value for eta_B  (not strictly required to be known, but it is easier to solve the first time increment starting with equilibrium value)
	const double eta_H_ref = mu_0_H + R * T * log(c_H_s_ref / c_H2O_s_ref)
	                       + mu_0_B  + R * T * log(c_B_s_ref / c_H2O_s_ref);		// initial equilibrium value for eta_H
	const double eta_OH_ref = mu_0_OH + R * T * log(c_OH_s_ref / c_H2O_s_ref)
	                        - mu_0_B  - R * T * log(c_B_s_ref / c_H2O_s_ref);		// initial equilibrium value for eta_OH (not strictly required to be known)
	const double phi_s_ref = -1.0 / z_B / F *
			                   (R * T * log(c_B_s_ref / c_H2O_s_ref) + mu_0_B);		// initial equilibrium value for phi in solution bath
	const double phi_g_ref = -1.0 / z_B / F *
			                   (R * T * log(c_B_g_ref / c_H2O_g_ref) + mu_0_B);		// initial equilibrium value for phi in gel
	const double U_th = - (eta_OH_ref + eta_H_ref) / 2.0 / F
			            + phi_c_ox - phi_c_red;										// threshold voltage to start water electrolysis

/*****************************************************
 * grid, assignment of domain and interface portions *
 *****************************************************/

	// make domain grid manually
	Triangulation<spacedim> tria_domain_0, tria_domain_1, tria_domain_2, tria_domain, tria_domain_ref;

	const double H_g = B_g * (double)N_H + 0.5 * B_g;
	const double H_s = B_g * ( (double)N_H  + 1.0 );
	const double B_s = B_g * ( 2.0 * (double)N_B + 1.0 );
	const double B_g_05 = 0.5 * B_g;
	const double B_s_05 = 0.5 * B_s;

	vector<unsigned int> repetitions = {2 * (1 + 2 * N_B), 2 * (N_H + 1)};
	Point<spacedim> p1(-B_s_05, 0.0);
	Point<spacedim> p2(B_s_05, H_s);
	GridGenerator::subdivided_hyper_rectangle(tria_domain_0, repetitions, p1, p2);
	set<typename Triangulation<spacedim>::active_cell_iterator> cells_to_remove;
	for(const auto& cell : tria_domain_0.active_cell_iterators())
	{
		if( (fabs(cell->center()[0]) < B_g_05) && (cell->center()[1] < H_g) )
			cells_to_remove.insert(cell);
	}
	GridGenerator::create_triangulation_with_removed_cells(tria_domain_0, cells_to_remove, tria_domain_1);
	const Point<spacedim> vertex_1(B_g_05, H_g);
	const Point<spacedim> vertex_2(-B_g_05, H_g);
	for(const auto& cell : tria_domain_1.cell_iterators_on_level(0))
	{
		for(unsigned int vertex = 0; vertex < GeometryInfo<spacedim>::vertices_per_cell; ++vertex)
		{
			if(cell->vertex(vertex).distance(vertex_1) < 1e-12)
			{
				cell->vertex(vertex)[0] -= (1.0 - sqrt(0.5)) * B_g * 0.5;
				cell->vertex(vertex)[1] -= (1.0 - sqrt(0.5)) * B_g * 0.5;
			}
			else if(cell->vertex(vertex).distance(vertex_2) < 1e-12)
			{
				cell->vertex(vertex)[0] += (1.0 - sqrt(0.5)) * B_g * 0.5;
				cell->vertex(vertex)[1] -= (1.0 - sqrt(0.5)) * B_g * 0.5;
			}

		}
	}
	const Point<spacedim> circle_center(0.0, 0.0);
	tria_domain_0.clear();
	GridGenerator::hyper_ball_balanced(tria_domain_0, circle_center, B_g_05);
	double shift_vertex_distance = 0.0;
	for(const auto& cell : tria_domain_0.cell_iterators_on_level(0))
	{
		for(unsigned int vertex = 0; vertex < GeometryInfo<spacedim>::vertices_per_cell; ++vertex)
			if(cell->vertex(vertex)[1] < 1e-12)
				if( (cell->vertex(vertex).distance(Point<spacedim>()) > 0.5 * B_g_05) && (cell->vertex(vertex).distance(Point<spacedim>()) < B_g_05 - 1e-12) )
				{
					shift_vertex_distance = cell->vertex(vertex).distance(Point<spacedim>());
					break;
				}
		if(shift_vertex_distance > 0.0)
			break;
	}
	cells_to_remove.clear();
	for(const auto& cell : tria_domain_0.active_cell_iterators())
	{
		if( cell->center()[1] < 0 )
			cells_to_remove.insert(cell);
	}
	GridGenerator::create_triangulation_with_removed_cells(tria_domain_0, cells_to_remove, tria_domain_2);
	Tensor<1, spacedim> shift_vector;
	shift_vector[1] = H_g - B_g_05;
	GridTools::shift(shift_vector, tria_domain_2);
	tria_domain_0.clear();
	GridGenerator::merge_triangulations(tria_domain_1, tria_domain_2, tria_domain_0);

	if(N_H > 0)
	{
		tria_domain_1.clear();
		tria_domain_2.clear();
		p1[0] = -B_g_05;
		p1[1] = 0.0;
		p2[0] = B_g_05;
		p2[1] = H_g - B_g_05;
		repetitions[0] = 4;
		repetitions[1] = 2 * N_H;
		GridGenerator::subdivided_hyper_rectangle(tria_domain_1, repetitions, p1, p2);
		for(const auto& cell : tria_domain_1.cell_iterators_on_level(0))
		{
			for(unsigned int vertex = 0; vertex < GeometryInfo<spacedim>::vertices_per_cell; ++vertex)
			{
				if(fabs(fabs(cell->vertex(vertex)[0]) - 0.5 * B_g_05) < 1e-12)
				{
					if(cell->vertex(vertex)[0] < 0)
						cell->vertex(vertex)[0] = -shift_vertex_distance;
					else
						cell->vertex(vertex)[0] = shift_vertex_distance;
				}
			}
		}
		GridGenerator::merge_triangulations(tria_domain_0, tria_domain_1, tria_domain);
	}
	else
		tria_domain.copy_triangulation(tria_domain_0);

	// define domain portions
	// 0 - hydrogel
	// 1 - solution
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		if( (fabs(cell->center()[0]) < B_g_05) && (cell->center()[1] < H_g) )
			cell->set_material_id(0);
		else
			cell->set_material_id(1);
	}

	// define manifolds
	tria_domain.set_all_manifold_ids(2);
	for(const auto& cell : tria_domain.active_cell_iterators())
	{
		for(unsigned int f = 0; f < GeometryInfo<spacedim>::faces_per_cell; ++f)
		{
			if(!cell->face(f)->at_boundary())
			{
				if( (cell->material_id() == 1) && (cell->neighbor(f)->material_id() == 0) && (cell->center()[1] > H_g - B_g_05) )
				{
					cell->face(f)->set_all_manifold_ids(1);
				}
			}
		}
	}
	tria_domain.set_all_manifold_ids_on_boundary(0);

	// make a copy for the error calculation
	tria_domain_ref.copy_triangulation(tria_domain);

	// triangulation system and interface definition (make also a copy for error calculation)
	// 0 - Sigma_el,l
	// 1 - Sigma_el,r
	// 2 - Sigma_g,Y=0
	// 3 - Sigma_s,Y=0
	// 5 - Sigma_s,Y=H_s
	// 6 - Sigma_sg
	TriangulationSystem<spacedim> tria_system(tria_domain);
	TriangulationSystem<spacedim> tria_system_ref(tria_domain_ref);

	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary())
			{
				if(cell->face(face)->center()[0] < -B_s_05 + 1e-12)
					tria_system.add_interface_cell(cell, face, 0);
				else if(cell->face(face)->center()[0] > B_s_05 - 1e-12)
					tria_system.add_interface_cell(cell, face, 1);
				else if(cell->face(face)->center()[1] <  1e-12)
				{
					if(fabs(cell->face(face)->center()[0]) < B_g_05)
						tria_system.add_interface_cell(cell, face, 2);
					else
						tria_system.add_interface_cell(cell, face, 3);
				}
				else if(cell->face(face)->center()[1] > H_s - 1e-12)
					tria_system.add_interface_cell(cell, face, 5);
			}
			else
			{
				if( (cell->material_id() == 1) && (cell->neighbor(face)->material_id() == 0) )
					tria_system.add_interface_cell(cell, face, 6);
			}
		}
	}

	for(const auto& cell : tria_domain_ref.cell_iterators_on_level(0))
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary())
			{
				if(cell->face(face)->center()[0] < -B_s_05 + 1e-12)
					tria_system_ref.add_interface_cell(cell, face, 0);
				else if(cell->face(face)->center()[0] > B_s_05 - 1e-12)
					tria_system_ref.add_interface_cell(cell, face, 1);
				else if(cell->face(face)->center()[1] <  1e-12)
				{
					if(fabs(cell->face(face)->center()[0]) < B_g_05)
						tria_system_ref.add_interface_cell(cell, face, 2);
					else
						tria_system_ref.add_interface_cell(cell, face, 3);
				}
				else if(cell->face(face)->center()[1] > H_s - 1e-12)
					tria_system_ref.add_interface_cell(cell, face, 5);
			}
			else
			{
				if( (cell->material_id() == 1) && (cell->neighbor(face)->material_id() == 0) )
					tria_system_ref.add_interface_cell(cell, face, 6);
			}
		}
	}

	// attach manifolds, so that curved interface of active particles is correctly represented upon mesh refinement
	SphericalManifold<spacedim> spherical_manifold_domain(Point<spacedim>(0.0, H_g - B_g_05));
	SphericalManifold<spacedim-1, spacedim> spherical_manifold_interface(Point<spacedim>(0.0, H_g - B_g_05));
	FlatManifold<spacedim> flat_manifold_domain;
	FlatManifold<spacedim-1, spacedim> flat_manifold_interface;
	TransfiniteInterpolationManifold<spacedim> transfinite_interpolation_manifold, transfinite_interpolation_manifold_ref;
	tria_domain.set_manifold(1, spherical_manifold_domain);
	tria_domain.set_manifold(0, flat_manifold_domain);
	transfinite_interpolation_manifold.initialize(tria_domain);
	tria_domain.set_manifold (2, transfinite_interpolation_manifold);
	tria_system.set_interface_manifold(1, spherical_manifold_interface);
	tria_system.set_interface_manifold(0, flat_manifold_interface);

	tria_domain_ref.set_manifold(1, spherical_manifold_domain);
	tria_domain_ref.set_manifold(0, flat_manifold_domain);
	transfinite_interpolation_manifold_ref.initialize(tria_domain_ref);
	tria_domain_ref.set_manifold (2, transfinite_interpolation_manifold_ref);
	tria_system_ref.set_interface_manifold(1, spherical_manifold_interface);
	tria_system_ref.set_interface_manifold(0, flat_manifold_interface);

	// finish definition of geometry
	tria_system.close();
	tria_system_ref.close();

	// mesh refinement
	const Point<spacedim> refine_vertex_l(-B_g_05/sqrt(2.0), H_g - B_g_05 + B_g_05/sqrt(2.0));
	const Point<spacedim> refine_vertex_r(B_g_05/sqrt(2.0), H_g - B_g_05 + B_g_05/sqrt(2.0));
	for(unsigned int m = 0; m < N_refinements_local_gs; ++m)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
			{
				if(!cell->face(face)->at_boundary())
				{
					if( (cell->material_id() == 0) && (cell->neighbor(face)->material_id() == 1) )
					{
						if(cell->face(face)->center()[1] < H_g - B_g_05)
						{
							if(cell->center()[0] > 0.0)
							{
								cell->set_refine_flag(RefinementCase<spacedim>::cut_y);
								cell->neighbor(face)->set_refine_flag(RefinementCase<spacedim>::cut_x);
							}
							else
							{
								cell->set_refine_flag(RefinementCase<spacedim>::cut_x);
								cell->neighbor(face)->set_refine_flag(RefinementCase<spacedim>::cut_x);
							}
						}
						else
						{
							cell->set_refine_flag();
							cell->neighbor(face)->set_refine_flag();
						}
					}
				}
			}
			if(cell->material_id() == 1)
			{
				for(unsigned int vertex = 0; vertex < GeometryInfo<spacedim>::vertices_per_cell; ++vertex)
					if( (cell->vertex(vertex).distance(refine_vertex_l) < 1e-12) || (cell->vertex(vertex).distance(refine_vertex_r) < 1e-12))
						cell->set_refine_flag();
			}
		}
		for(const auto& cell : tria_domain_ref.active_cell_iterators())
		{
			for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
			{
				if(!cell->face(face)->at_boundary())
				{
					if( (cell->material_id() == 0) && (cell->neighbor(face)->material_id() == 1) )
					{
						if(cell->face(face)->center()[1] < H_g - B_g_05)
						{
							if(cell->center()[0] > 0.0)
							{
								cell->set_refine_flag(RefinementCase<spacedim>::cut_y);
								cell->neighbor(face)->set_refine_flag(RefinementCase<spacedim>::cut_x);
							}
							else
							{
								cell->set_refine_flag(RefinementCase<spacedim>::cut_x);
								cell->neighbor(face)->set_refine_flag(RefinementCase<spacedim>::cut_x);
							}
						}
						else
						{
							cell->set_refine_flag();
							cell->neighbor(face)->set_refine_flag();
						}
					}
				}
			}
			if(cell->material_id() == 1)
			{
				for(unsigned int vertex = 0; vertex < GeometryInfo<spacedim>::vertices_per_cell; ++vertex)
					if( (cell->vertex(vertex).distance(refine_vertex_l) < 1e-12) || (cell->vertex(vertex).distance(refine_vertex_r) < 1e-12))
						cell->set_refine_flag();
			}
		}
		tria_domain.execute_coarsening_and_refinement();
		tria_domain_ref.execute_coarsening_and_refinement();
	}

	for(unsigned int m = 0; m < N_refinements_local_el; ++m)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
			for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
				if( (cell->face(face)->at_boundary()) && (fabs(cell->face(face)->center()[0]) > B_s_05 - 1e-12) )
					cell->set_refine_flag(RefinementCase<spacedim>::cut_x);
		for(const auto& cell : tria_domain_ref.active_cell_iterators())
			for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
				if( (cell->face(face)->at_boundary()) && (fabs(cell->face(face)->center()[0]) > B_s_05 - 1e-12) )
					cell->set_refine_flag(RefinementCase<spacedim>::cut_x);
		tria_domain.execute_coarsening_and_refinement();
		tria_domain_ref.execute_coarsening_and_refinement();
	}

	tria_domain.refine_global(N_refinements_global);
	tria_domain_ref.refine_global(m_h_reference);

/**************************************
 * unknowns and Dirichlet constraints *
 **************************************/

	ConstantFunctionPiecewise<spacedim, spacedim> p_initial(p_g_ref, p_s_ref);									// initial condition for pressure
	ConstantFunctionPiecewise<spacedim, spacedim> phi_ref_(phi_g_ref, phi_s_ref);								// initial condition for pressure

	ConstantFunctionPiecewise<spacedim, spacedim> c_H2O_initial(c_H2O_g_ref, c_H2O_s_ref);						// initial condition for water concentration
	ConstantFunctionPiecewise<spacedim, spacedim> c_A_initial(c_A_g_ref, c_A_s_ref);							// initial condition for concentration of A+ ions
	ConstantFunctionPiecewise<spacedim, spacedim> c_B_initial(c_B_g_ref, c_B_s_ref);							// initial condition for concentration of B- ions
	Functions::ConstantFunction<spacedim> c_C_initial(c_C_ref);													// initial condition for concentration of fixed charged groups in gel
	ConstantFunctionPiecewise<spacedim, spacedim> c_H_initial(c_H_g_ref, c_H_s_ref);							// initial condition for concentration of H+ ions
	ConstantFunctionPiecewise<spacedim, spacedim> c_OH_initial(c_OH_g_ref, c_OH_s_ref);							// initial condition for concentration of OH- ions
	Functions::ConstantFunction<spacedim> eta_A_initial(eta_A_ref);												// initial value for eta_A
	Functions::ConstantFunction<spacedim> eta_B_initial(eta_B_ref);												// initial value for eta_B
	Functions::ConstantFunction<spacedim> eta_H_initial(eta_H_ref);												// initial value for eta_H
	Functions::ConstantFunction<spacedim> eta_OH_initial(eta_OH_ref);											// initial value for eta_OH

	IndependentField<spacedim, spacedim> u("u", FE_Q<spacedim>(degree), spacedim, {0,1});																		// displacement field of solid
	IndependentField<spacedim, spacedim> c_H2O("c_H2O",  FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0,1}, &c_H2O_initial, use_local_elimination);	// concentration of water
	IndependentField<spacedim, spacedim> c_A( "c_A",  FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0,1}, &c_A_initial, use_local_elimination);		// concentration of A+ ions
	IndependentField<spacedim, spacedim> c_B( "c_B",  FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0,1}, &c_B_initial, use_local_elimination);		// concentration of B- ions
	IndependentField<spacedim, spacedim> c_H( "c_H",  FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0,1}, &c_H_initial, use_local_elimination);		// concentration of H+ ions
	IndependentField<spacedim, spacedim> c_OH("c_OH", FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0,1}, &c_OH_initial, use_local_elimination);		// concentration of OH- ions
	IndependentField<spacedim, spacedim> c_C("c_C",   FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0}, &c_C_initial, use_local_elimination);		// concentration of fixed charged groups in gel (only needed if dissociation is taken into account)
	IndependentField<0, spacedim> u_N("u_N");																													// Constant normal displacement of the fictitious solid at the top boundary of the domain
	IndependentField<spacedim, spacedim> p("p", FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0,1}, &p_initial, use_local_elimination);				// pressure field
	IndependentField<spacedim, spacedim> phi("phi", FE_DGQArbitraryNodes<spacedim>(QGauss<1>(degree+1)), 1, {0,1}, &phi_ref_, use_local_elimination);			// scalar potential
	IndependentField<spacedim, spacedim> l_s("l_s", FE_Q<spacedim>(degree), spacedim, {1});																		// auxiliary field for eliminating arbitrariness from displacement solution on solution bath domain
	IndependentField<0, spacedim> l_u_x("C_u_x_avg");																											// independent field for constraining the average lateral displacement of the gel to zero

	IndependentField<spacedim, spacedim> v_s("v_s", FE_Q<spacedim>(degree+1), spacedim, {1});								// time integrated water velocity in solution bath (only needed if with_infinitesimal_viscosity==true)
	IndependentField<spacedim, spacedim> xi_s( "xi_s", FE_Q<spacedim>(degree), 1, {1});										// flow potential if with_infinitesimal_viscosity==false, process variable for implementing zero viscosity limit else
	IndependentField<spacedim-1, spacedim> t_S("t_S", FE_Q<spacedim-1, spacedim>(degree), 1, {0,1,3,5,6});					// process variable for implementing zero viscosity limit (only needed if with_infinitesimal_viscosity==true)
	IndependentField<spacedim-1, spacedim> t_t_S("t_t_S", FE_Q<spacedim-1, spacedim>(degree+1), 1, {0,1,5,6});				// process variable for implementing zero viscosity limit (only needed if with_infinitesimal_viscosity==true)

	IndependentField<spacedim, spacedim> eta_A( "eta_A",  FE_Q<spacedim>(degree), 1, {0,1}, &eta_A_initial);				// electrochemical potential of A+ ions in gel and solution bath
	IndependentField<spacedim, spacedim> eta_B( "eta_B",  FE_Q<spacedim>(degree), 1, {0,1}, &eta_B_initial);				// electrochemical potential of B- ions in gel and solution bath
	IndependentField<spacedim, spacedim> eta_H( "eta_H",  FE_Q<spacedim>(degree), 1, {0,1}, &eta_H_initial);				// electrochemical potential of H+ ions in gel and solution bath
	IndependentField<spacedim, spacedim> eta_OH("eta_OH", FE_Q<spacedim>(degree), 1, {0,1}, &eta_OH_initial);				// electrochemical potential of OH- ions in gel and solution bath
	IndependentField<spacedim, spacedim> f_s("f_s", FE_Q<spacedim>(degree+1), spacedim, {1});								// Lagrange multiplier for implementing zero viscosity limit (only needed if with_infinitesimal_viscosity==true)
	IndependentField<spacedim-1, spacedim> eta_S("eta_S", FE_Q<spacedim-1, spacedim>(degree), 1, {0,1,3,5,6});				// Lagrange multiplier for implementing zero viscosity limit (only needed if with_infinitesimal_viscosity==true)
	IndependentField<spacedim-1, spacedim> eta_t_S("eta_t_S", FE_Q<spacedim-1, spacedim>(degree+1), 1, {0,1,5,6});			// Lagrange multiplier constraining tangential velocity to zero (only needed if with_infinitesimal_viscosity==true)

	IndependentField<0, spacedim> phi_e("phi_e",  -phi_c_red - eta_H_ref / F);												// Lagrangian multiplier for balancing the electron flow

	// define constraints for function spaces
	DirichletConstraint<spacedim> dc_u_x_left_right(u, 0, InterfaceSide::minus, {0,1});										// normal displacement left and right side
	DirichletConstraint<spacedim> dc_u_y_bottom(u, 1, InterfaceSide::minus, {2,3});											// normal displacement bottom
	DirichletConstraint<spacedim> dc_u_y_top(u, 1, InterfaceSide::minus, {5}, nullptr, &u_N);								// normal displacement top (constant)
	DirichletConstraint<spacedim> dc_l_s_x(l_s, 0, InterfaceSide::minus, {0,1,6});											// normal artificial displacement left side, right side, artificial displacement in X-direction solution bath/gel interface
	DirichletConstraint<spacedim> dc_l_s_y(l_s, 1, InterfaceSide::minus, {3,5,6});											// normal artificial displacement bottom, top, artificial displacement in Y-direction solution bath/gel interface
	PointConstraint<spacedim, spacedim> dc_eta_B(eta_B, 0, Point<spacedim>(0.0, H_s));										// constrain constant of eta_B
	PointConstraint<spacedim, spacedim> dc_xi_s(xi_s, 0, Point<spacedim>(0.0, H_s));										// constrain constant of xi_s

	// finally assemble the constraints into the constraints object
	Constraints<spacedim> constraints;
	constraints.add_dirichlet_constraint(dc_u_x_left_right);
	constraints.add_dirichlet_constraint(dc_u_y_bottom);
	constraints.add_dirichlet_constraint(dc_u_y_top);
	constraints.add_dirichlet_constraint(dc_l_s_x);
	constraints.add_dirichlet_constraint(dc_l_s_y);
	constraints.add_point_constraint(dc_eta_B);
	constraints.add_point_constraint(dc_xi_s);

/********************
 * dependent fields *
 ********************/

	// deformation gradient
	DependentField<spacedim, spacedim> F_xx("F_xx");
	DependentField<spacedim, spacedim> F_yy("F_yy");
	DependentField<spacedim, spacedim> F_zz("F_zz");
	DependentField<spacedim, spacedim> F_xy("F_xy");
	DependentField<spacedim, spacedim> F_yz("F_yz");
	DependentField<spacedim, spacedim> F_zx("F_zx");
	DependentField<spacedim, spacedim> F_yx("F_yx");
	DependentField<spacedim, spacedim> F_zy("F_zy");
	DependentField<spacedim, spacedim> F_xz("F_xz");
	F_xx.add_term(1.0, u, 0, 0);
	F_xx.add_term(1.0);
	F_yy.add_term(1.0, u, 1, 1);
	F_yy.add_term(1.0);
	F_zz.add_term(1.0);
	F_xy.add_term(1.0, u, 0, 1);
	F_yx.add_term(1.0, u, 1, 0);

	DependentField<spacedim-1, spacedim> F_xx_minus("F_xx");
	DependentField<spacedim-1, spacedim> F_yy_minus("F_yy");
	DependentField<spacedim-1, spacedim> F_zz_minus("F_zz");
	DependentField<spacedim-1, spacedim> F_xy_minus("F_xy");
	DependentField<spacedim-1, spacedim> F_yz_minus("F_yz");
	DependentField<spacedim-1, spacedim> F_zx_minus("F_zx");
	DependentField<spacedim-1, spacedim> F_yx_minus("F_yx");
	DependentField<spacedim-1, spacedim> F_zy_minus("F_zy");
	DependentField<spacedim-1, spacedim> F_xz_minus("F_xz");
	F_xx_minus.add_term(1.0, u, 0, 0, InterfaceSide::minus);
	F_xx_minus.add_term(1.0);
	F_yy_minus.add_term(1.0, u, 1, 1, InterfaceSide::minus);
	F_yy_minus.add_term(1.0);
	F_zz_minus.add_term(1.0);
	F_xy_minus.add_term(1.0, u, 0, 1, InterfaceSide::minus);
	F_yx_minus.add_term(1.0, u, 1, 0, InterfaceSide::minus);

	// displacements
	DependentField<spacedim, spacedim> u_x("u_x");
	DependentField<spacedim, spacedim> u_y("u_y");
	DependentField<spacedim, spacedim> u_z("u_z");
	u_x.add_term(1.0, u, 0);
	u_y.add_term(1.0, u, 1);

	DependentField<spacedim-1, spacedim> u_x_minus("u_x");
	DependentField<spacedim-1, spacedim> u_y_minus("u_y");
	DependentField<spacedim-1, spacedim> u_z_minus("u_z");
	u_x_minus.add_term(1.0, u, 0, InterfaceSide::minus);
	u_y_minus.add_term(1.0, u, 1, InterfaceSide::minus);

	// water concentration
	DependentField<spacedim, spacedim> c_H2O_("c_H2O");
	c_H2O_.add_term(1.0, c_H2O);

	// A+ ion concentration
	DependentField<spacedim, spacedim> c_A_("c_A");
	c_A_.add_term(1.0, c_A);

	// B- ion concentration
	DependentField<spacedim, spacedim> c_B_("c_B");
	c_B_.add_term(1.0, c_B);

	// H+ ion concentration
	DependentField<spacedim, spacedim> c_H_("c_H");
	c_H_.add_term(1.0, c_H);

	// OH- ion concentration
	DependentField<spacedim, spacedim> c_OH_("c_OH");
	c_OH_.add_term(1.0, c_OH);

	// C- ion concentration
	DependentField<spacedim, spacedim> c_C_("c_C");
	if(with_CH_dissociation)
		c_C_.add_term(1.0, c_C);
	else
		c_C_.add_term(c_C_ref);

	// p
	DependentField<spacedim, spacedim> p_("p");
	p_.add_term(1.0, p);

	// phi
	DependentField<spacedim, spacedim> phi_("phi");
	phi_.add_term(1.0, phi);

	// gradient of l_s
	DependentField<spacedim, spacedim> l_s_xx("l_s_xx");
	DependentField<spacedim, spacedim> l_s_yy("l_s_yy");
	DependentField<spacedim, spacedim> l_s_zz("l_s_zz");
	DependentField<spacedim, spacedim> l_s_xy("l_s_xy");
	DependentField<spacedim, spacedim> l_s_yz("l_s_yz");
	DependentField<spacedim, spacedim> l_s_zx("l_s_zx");
	DependentField<spacedim, spacedim> l_s_yx("l_s_yx");
	DependentField<spacedim, spacedim> l_s_zy("l_s_zy");
	DependentField<spacedim, spacedim> l_s_xz("l_s_xz");
	l_s_xx.add_term(1.0, l_s, 0, 0);
	l_s_yy.add_term(1.0, l_s, 1, 1);
	l_s_xy.add_term(1.0, l_s, 0, 1);
	l_s_yx.add_term(1.0, l_s, 1, 0);

	// l_u_x_
	DependentField<spacedim, spacedim> l_u_x_("l_u_x");
	l_u_x_.add_term(1.0, l_u_x);

	// water velocity
	DependentField<spacedim, spacedim> v_s_x("v_s_x");
	DependentField<spacedim, spacedim> v_s_y("v_s_y");
	DependentField<spacedim, spacedim> v_s_z("v_s_z");
	v_s_x.add_term(1.0, v_s, 0);
	v_s_y.add_term(1.0, v_s, 1);

	DependentField<spacedim-1, spacedim> v_s_x_minus("v_s_x");
	DependentField<spacedim-1, spacedim> v_s_y_minus("v_s_y");
	DependentField<spacedim-1, spacedim> v_s_z_minus("v_s_z");
	v_s_x_minus.add_term(1.0, v_s, 0, InterfaceSide::minus);
	v_s_y_minus.add_term(1.0, v_s, 1, InterfaceSide::minus);

	// gradient of water velocity
	DependentField<spacedim, spacedim> v_s_xx("v_s_xx");
	DependentField<spacedim, spacedim> v_s_xy("v_s_xy");
	DependentField<spacedim, spacedim> v_s_xz("v_s_xz");
	DependentField<spacedim, spacedim> v_s_yx("v_s_yx");
	DependentField<spacedim, spacedim> v_s_yy("v_s_yy");
	DependentField<spacedim, spacedim> v_s_yz("v_s_yz");
	DependentField<spacedim, spacedim> v_s_zx("v_s_zx");
	DependentField<spacedim, spacedim> v_s_zy("v_s_zy");
	DependentField<spacedim, spacedim> v_s_zz("v_s_zz");
	v_s_xx.add_term(1.0, v_s, 0, 0);
	v_s_yy.add_term(1.0, v_s, 1, 1);
	v_s_xy.add_term(1.0, v_s, 0, 1);
	v_s_yx.add_term(1.0, v_s, 1, 0);

	DependentField<spacedim-1, spacedim> v_s_xx_minus("v_s_xx");
	DependentField<spacedim-1, spacedim> v_s_xy_minus("v_s_xy");
	DependentField<spacedim-1, spacedim> v_s_xz_minus("v_s_xz");
	DependentField<spacedim-1, spacedim> v_s_yx_minus("v_s_yx");
	DependentField<spacedim-1, spacedim> v_s_yy_minus("v_s_yy");
	DependentField<spacedim-1, spacedim> v_s_yz_minus("v_s_yz");
	DependentField<spacedim-1, spacedim> v_s_zx_minus("v_s_zx");
	DependentField<spacedim-1, spacedim> v_s_zy_minus("v_s_zy");
	DependentField<spacedim-1, spacedim> v_s_zz_minus("v_s_zz");
	v_s_xx_minus.add_term(1.0, v_s, 0, 0, InterfaceSide::minus);
	v_s_yy_minus.add_term(1.0, v_s, 1, 1, InterfaceSide::minus);
	v_s_xy_minus.add_term(1.0, v_s, 0, 1, InterfaceSide::minus);
	v_s_yx_minus.add_term(1.0, v_s, 1, 0, InterfaceSide::minus);

	// gradient of xi_s
	DependentField<spacedim, spacedim> xi_s_x("xi_s_x");
	DependentField<spacedim, spacedim> xi_s_y("xi_s_y");
	DependentField<spacedim, spacedim> xi_s_z("xi_s_z");
	xi_s_x.add_term(1.0, xi_s, 0, 0);
	xi_s_y.add_term(1.0, xi_s, 0, 1);

	// t_S
	DependentField<spacedim-1, spacedim> t_S_("t_S_");
	t_S_.add_term(1.0, t_S, 0);

	// t_t_S
	DependentField<spacedim-1, spacedim> t_t_S_x("t_t_S_x");
	DependentField<spacedim-1, spacedim> t_t_S_y("t_t_S_y");
	DependentField<spacedim-1, spacedim> t_t_S_z("t_t_S_z");
	t_t_S_z.add_term(1.0, t_t_S, 0);

	// eta_A
	DependentField<spacedim, spacedim> eta_A_("eta_A");
	eta_A_.add_term(1.0, eta_A);

	// gradient of eta_A
	DependentField<spacedim, spacedim> eta_A_x("eta_A_x");
	DependentField<spacedim, spacedim> eta_A_y("eta_A_y");
	DependentField<spacedim, spacedim> eta_A_z("eta_A_z");
	eta_A_x.add_term(1.0, eta_A, 0, 0);
	eta_A_y.add_term(1.0, eta_A, 0, 1);

	// eta_B
	DependentField<spacedim, spacedim> eta_B_("eta_B");
	eta_B_.add_term(1.0, eta_B);

	// gradient of eta_B
	DependentField<spacedim, spacedim> eta_B_x("eta_B_x");
	DependentField<spacedim, spacedim> eta_B_y("eta_B_y");
	DependentField<spacedim, spacedim> eta_B_z("eta_B_z");
	eta_B_x.add_term(1.0, eta_B, 0, 0);
	eta_B_y.add_term(1.0, eta_B, 0, 1);

	// eta_H
	DependentField<spacedim, spacedim> eta_H_("eta_H");
	eta_H_.add_term(1.0, eta_H);

	// gradient of eta_H
	DependentField<spacedim, spacedim> eta_H_x("eta_H_x");
	DependentField<spacedim, spacedim> eta_H_y("eta_H_y");
	DependentField<spacedim, spacedim> eta_H_z("eta_H_z");
	eta_H_x.add_term(1.0, eta_H, 0, 0);
	eta_H_y.add_term(1.0, eta_H, 0, 1);

	// eta_OH
	DependentField<spacedim, spacedim> eta_OH_("eta_OH");
	eta_OH_.add_term(1.0, eta_OH);

	// gradient of eta_OH
	DependentField<spacedim, spacedim> eta_OH_x("eta_OH_x");
	DependentField<spacedim, spacedim> eta_OH_y("eta_OH_y");
	DependentField<spacedim, spacedim> eta_OH_z("eta_OH_z");
	eta_OH_x.add_term(1.0, eta_OH, 0, 0);
	eta_OH_y.add_term(1.0, eta_OH, 0, 1);

	// gradient of f_s
	DependentField<spacedim, spacedim> f_s_xx("f_s_xx");
	DependentField<spacedim, spacedim> f_s_xy("f_s_xy");
	DependentField<spacedim, spacedim> f_s_xz("f_s_xz");
	DependentField<spacedim, spacedim> f_s_yx("f_s_yx");
	DependentField<spacedim, spacedim> f_s_yy("f_s_yy");
	DependentField<spacedim, spacedim> f_s_yz("f_s_yz");
	DependentField<spacedim, spacedim> f_s_zx("f_s_zx");
	DependentField<spacedim, spacedim> f_s_zy("f_s_zy");
	DependentField<spacedim, spacedim> f_s_zz("f_s_zz");
	f_s_xx.add_term(1.0, f_s, 0, 0);
	f_s_yy.add_term(1.0, f_s, 1, 1);
	f_s_xy.add_term(1.0, f_s, 0, 1);
	f_s_yx.add_term(1.0, f_s, 1, 0);

	DependentField<spacedim-1, spacedim> f_s_xx_minus("f_s_xx");
	DependentField<spacedim-1, spacedim> f_s_xy_minus("f_s_xy");
	DependentField<spacedim-1, spacedim> f_s_xz_minus("f_s_xz");
	DependentField<spacedim-1, spacedim> f_s_yx_minus("f_s_yx");
	DependentField<spacedim-1, spacedim> f_s_yy_minus("f_s_yy");
	DependentField<spacedim-1, spacedim> f_s_yz_minus("f_s_yz");
	DependentField<spacedim-1, spacedim> f_s_zx_minus("f_s_zx");
	DependentField<spacedim-1, spacedim> f_s_zy_minus("f_s_zy");
	DependentField<spacedim-1, spacedim> f_s_zz_minus("f_s_zz");
	f_s_xx_minus.add_term(1.0, f_s, 0, 0, InterfaceSide::minus);
	f_s_yy_minus.add_term(1.0, f_s, 1, 1, InterfaceSide::minus);
	f_s_xy_minus.add_term(1.0, f_s, 0, 1, InterfaceSide::minus);
	f_s_yx_minus.add_term(1.0, f_s, 1, 0, InterfaceSide::minus);

	// f_s
	DependentField<spacedim, spacedim> f_s_x("f_s_x");
	DependentField<spacedim, spacedim> f_s_y("f_s_y");
	DependentField<spacedim, spacedim> f_s_z("f_s_z");
	f_s_x.add_term(1.0, f_s, 0);
	f_s_y.add_term(1.0, f_s, 1);

	DependentField<spacedim-1, spacedim> f_s_x_minus("f_s_x");
	DependentField<spacedim-1, spacedim> f_s_y_minus("f_s_y");
	DependentField<spacedim-1, spacedim> f_s_z_minus("f_s_z");
	f_s_x_minus.add_term(1.0, f_s, 0, InterfaceSide:: minus);
	f_s_y_minus.add_term(1.0, f_s, 1, InterfaceSide:: minus);

	// eta_S
	DependentField<spacedim-1, spacedim> eta_S_("eta_S_");
	eta_S_.add_term(1.0, eta_S, 0);

	// eta_t_S
	DependentField<spacedim-1, spacedim> eta_t_S_x("eta_t_S_x");
	DependentField<spacedim-1, spacedim> eta_t_S_y("eta_t_S_y");
	DependentField<spacedim-1, spacedim> eta_t_S_z("eta_t_S_z");
	eta_t_S_z.add_term(1.0, eta_t_S, 0);

	// eta_H2O
	DependentField<spacedim, spacedim> eta_H2O_("eta_H2O");
	eta_H2O_.add_term(1.0, eta_H);
	eta_H2O_.add_term(1.0, eta_OH);

	// gradient of eta_H2O
	DependentField<spacedim, spacedim> eta_H2O_x("eta_H2O_x");
	DependentField<spacedim, spacedim> eta_H2O_y("eta_H2O_y");
	DependentField<spacedim, spacedim> eta_H2O_z("eta_H2O_z");
	eta_H2O_x.add_term(1.0, eta_H, 0, 0);
	eta_H2O_x.add_term(1.0, eta_OH, 0, 0);
	eta_H2O_y.add_term(1.0, eta_H, 0, 1);
	eta_H2O_y.add_term(1.0, eta_OH, 0, 1);

	// internal potential reduction
	DependentField<spacedim-1, spacedim> delta_eta_0_red("delta_eta_0_red");
	delta_eta_0_red.add_term(1.0, eta_H, 0, InterfaceSide::minus);
	delta_eta_0_red.add_term(A_e_red * F, phi_e);

	// internal potential oxidation
	DependentField<spacedim-1, spacedim> delta_eta_0_ox("delta_eta_0_ox");
	delta_eta_0_ox.add_term(1.0, eta_OH, 0, InterfaceSide::minus);
	delta_eta_0_ox.add_term(-1.0, eta_H, 0, InterfaceSide::minus);
	delta_eta_0_ox.add_term(A_e_ox * F, phi_e);

	// eta_C
	DependentField<spacedim, spacedim> eta_C("eta_C");
	if(with_CH_dissociation)
		eta_C.add_term(-1.0, eta_H);
	else
		eta_C.add_term(-eta_H_ref);

/*************************
 * incremental potential *
 *************************/

	// Note: The actual implementations of the forms of the contributions to the total potential (in particular, the integrands of the integrals) are contained in
	// the following global libraries:
	// 		incremental_fe/scalar_functionals/psi_lib.h (for contributions to the free energy Psi)
	//		incremental_fe/scalar_functionals/omega_lib.h (for contributions to Omega),
	// so that these can be re-used for the modeling of other problems.
	//
	// Note: "Omega" corresponds to "Gamma" in the manuscript (the symbol "Gamma" has been used in the manuscript in order to avoid a duplicate with the symbol used for the domain;
	//                                                         however, this name duplication has not yet been eliminated from the libraries)

	// strain energy of polymeric backbone in gel - psi^U, part 1
	Functions::ConstantFunction<spacedim> constant_function(1.0);
	PsiNeoHooke00<spacedim> psi_U_1({F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									lambda_p,
									mu_p,
									constant_function,
									alpha,
									nullptr,
									1.0/n_p_ref);

	// free energy of mixing of polymer and water - psi^U, part 2
	PsiChemical07<spacedim> psi_U_2({c_H2O_},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									R*T,
									n_p_ref,
									V_m_H2O,
									chi,
									alpha);

	// chemical free energy related to water in gel - psi^U, part 3
	FullMatrix<double> C(1);
	Vector<double> y(1);
	y[0] = psi_c_0;
	PsiLinear00<spacedim> psi_U_3(	{c_H2O_},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									C,
									y,
									alpha);

	// chemical free energy related to A+ ions in gel - psi^U, part 4
	PsiChemical02<spacedim> psi_U_4({c_A_, c_H2O_},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									R*T,
									mu_0_A,
									alpha,
									eps,
									c_A_g_ref / c_H2O_g_ref);

	// chemical free energy related to B- ions in gel - psi^U, part 5
	PsiChemical02<spacedim> psi_U_5({c_B_, c_H2O_},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									R*T,
									mu_0_B,
									alpha,
									eps,
									c_B_g_ref / c_H2O_g_ref);

	// chemical free energy related to H+ ions in gel - psi^U, part 6
	PsiChemical02<spacedim> psi_U_6({c_H_, c_H2O_},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									R*T,
									mu_0_H,
									alpha,
									eps,
									c_H_g_ref / c_H2O_g_ref);

	// chemical free energy related to OH- ions in gel - psi^U, part 7
	PsiChemical02<spacedim> psi_U_7(	{c_OH_, c_H2O_},
										{0},
										QGauss<spacedim>(degree+1),
										global_data,
										R*T,
										mu_0_OH,
										alpha,
										eps,
										c_OH_g_ref / c_H2O_g_ref);


	// chemical free energy related to water in solution bath - psi^U, part 8
	C.reinit(1,1);
	y.reinit(1);
	y[0] = psi_c_0;
	PsiLinear00<spacedim> psi_U_8(	{c_H2O_},
									{1},
									QGauss<spacedim>(degree+1),
									global_data,
									C,
									y,
									alpha);

	// chemical free energy related to A+ ions in solution bath - psi^U, part 9
	PsiChemical02<spacedim> psi_U_9({c_A_, c_H2O_},
									{1},
									QGauss<spacedim>(degree+1),
									global_data,
									R*T,
									mu_0_A,
									alpha,
									eps,
									c_A_s_ref / c_H2O_s_ref);

	// chemical free energy related to B- ions in solution bath - psi^U, part 10
	PsiChemical02<spacedim> psi_U_10(	{c_B_, c_H2O_},
										{1},
										QGauss<spacedim>(degree+1),
										global_data,
										R*T,
										mu_0_B,
										alpha,
										eps,
										c_B_s_ref / c_H2O_s_ref);

	// chemical free energy related to H+ ions in solution bath - psi^U, part 11
	PsiChemical02<spacedim> psi_U_11(	{c_H_, c_H2O_},
										{1},
										QGauss<spacedim>(degree+1),
										global_data,
										R*T,
										mu_0_H,
										alpha,
										eps,
										c_H_s_ref / c_H2O_s_ref);

	// chemical free energy related to OH- ions in solution bath - psi^U, part 12
	PsiChemical02<spacedim> psi_U_12(	{c_OH_, c_H2O_},
										{1},
										QGauss<spacedim>(degree+1),
										global_data,
										R*T,
										mu_0_OH,
										alpha,
										eps,
										c_OH_s_ref / c_H2O_s_ref);


	// chemical free energy related to C- groups of polymeric backbone, used only if dissociation of CH groups is taken into account - psi^U, part 13
	PsiChemical00<spacedim> psi_U_13(	{c_C_},
										{0},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T,
										c_C_ref,
										delta_mu_CH + R*T*log(c_C_ref/(c_CH_ref + c_C_ref)),
										alpha,
										eps);

	// chemical free energy related to CH groups of polymeric backbone, used only if dissociation of CH groups is taken into account - psi^U, part 14
	PsiChemical00<spacedim> psi_U_14(	{c_C_},
										{0},
										QGauss<spacedim>(degree + 1),
										global_data,
										R*T,
										c_CH_ref,
										R*T*log(c_CH_ref/(c_CH_ref + c_C_ref)),
										alpha,
										eps,
										-1.0,
										c_CH_ref + c_C_ref);


	// incompressibility constraint in gel - psi^P, part 1
	PsiIncompressibility00<spacedim> psi_P_1(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_H2O_, p_},
												{0},
												QGauss<spacedim>(degree+1),
												global_data,
												V_m_H2O,
												n_p_ref,
												false,
												false,
												alpha);

	// incompressibility constraint in solution bath - psi^P, part 2
	PsiIncompressibility00<spacedim> psi_P_2(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_H2O_, p_},
												{1},
												QGauss<spacedim>(degree+1),
												global_data,
												V_m_H2O,
												0.0,
												false,
												false,
												alpha);

	// electroneutrality constraint in gel - psi^P, part 3
	C.reinit(6,6);
	y.reinit(6);
	C(0,5) = C(5,0) = z_A * F;
	C(1,5) = C(5,1) = z_B * F;
	C(2,5) = C(5,2) = z_H * F;
	C(3,5) = C(5,3) = z_OH * F;
	C(4,5) = C(5,4) = z_C * F;
	PsiLinear00<spacedim> psi_P_3(	{c_A_, c_B_, c_H_, c_OH_, c_C_, phi_},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									C,
									y,
									alpha);

	// electroneutrality constraint in solution bath - psi^P, part 4
	C.reinit(5,5);
	y.reinit(5);
	C(0,4) = C(4,0) = z_A * F;
	C(1,4) = C(4,1) = z_B * F;
	C(2,4) = C(4,2) = z_H * F;
	C(3,4) = C(4,3) = z_OH * F;
	PsiLinear00<spacedim> psi_P_4(	{c_A_, c_B_, c_H_, c_OH_, phi_},
									{1},
									QGauss<spacedim>(degree+1),
									global_data,
									C,
									y,
									alpha);

	// constrain average x-displacement of hydrogel body to zero - psi^P, part 5
	C.reinit(2,2);
	y.reinit(2);
	C(0,1) = C(1,0) = 1.0;
	PsiLinear00<spacedim> psi_P_5(	{u_x, l_u_x_},
									{0},
									QGauss<spacedim>(degree+1),
									global_data,
									C,
									y,
									alpha);

	// constraint on displacement states in solution bath - psi^P, part 6
	LameScaling<spacedim> lame_scaling(B_g_05, 10.0);
	PsiNeoHookeLagrange00<spacedim> psi_P_6({F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, l_s_xx, l_s_xy, l_s_xz, l_s_yx, l_s_yy, l_s_yz, l_s_zx, l_s_zy, l_s_zz},
											{1},
											QGauss<spacedim>(degree+1),
											global_data,
											lambda_s,
											mu_s,
											lame_scaling,
											alpha);


	// part 1 of omega (related to water flux in gel)
	OmegaDualFluidDissipation00<spacedim> omega_1({
													c_H2O_,
													eta_H2O_x,  eta_H2O_y,  eta_H2O_z,
													eta_A_x,  eta_A_y,	eta_A_z,
													eta_B_x,  eta_B_y,  eta_B_z,
													eta_OH_x, eta_OH_y,	eta_OH_z,
													eta_H_x,  eta_H_y,  eta_H_z,
													eta_H2O_,
													c_H2O_,
													c_A_,
													c_B_,
													c_OH_,
													c_H_,
													F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz
													},
													{0},
													QGauss<spacedim>(degree+1),
													global_data,
													4,
													D_H2O / (R*T),
													V_m_H2O,
													method,
													alpha);

	// part 2 of omega (related to flux of A+ in gel)
	OmegaDualIonDissipation00<spacedim> omega_2({c_A_, eta_A_x, eta_A_y, eta_A_z, eta_A_, c_A_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
												{0},
												QGauss<spacedim>(degree+1),
												global_data,
												D_A / (R*T),
												V_m_H2O,
												method,
												alpha);

	// part 3 of omega (related to flux of B- in gel)
	OmegaDualIonDissipation00<spacedim> omega_3({c_B_, eta_B_x, eta_B_y, eta_B_z, eta_B_, c_B_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
												{0},
												QGauss<spacedim>(degree+1),
												global_data,
												D_B / (R*T),
												V_m_H2O,
												method,
												alpha);

	// part 4 of omega (related to flux of H+ in gel)
	OmegaDualIonDissipation00<spacedim> omega_4({c_H_, eta_H_x, eta_H_y, eta_H_z, eta_H_, c_H_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
												{0},
												QGauss<spacedim>(degree+1),
												global_data,
												D_H / (R*T),
												V_m_H2O,
												method,
												alpha);

	// part 5 of omega (related to flux of OH- in gel)
	OmegaDualIonDissipation00<spacedim> omega_5({c_OH_, eta_OH_x, eta_OH_y, eta_OH_z, eta_OH_, c_OH_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
												{0},
												QGauss<spacedim>(degree+1),
												global_data,
												D_OH / (R*T),
												V_m_H2O,
												method,
												alpha);

	// part 6 of omega (related to water flux in solution bath), used in case that velocity field is derived from potential
	OmegaDualFluidDissipation04<spacedim> omega_6(	{
													xi_s_x, xi_s_y, xi_s_z,
													c_H2O_,
													u_x, u_y, u_z,
													eta_H2O_x,  eta_H2O_y,  eta_H2O_z,
													eta_A_x,  eta_A_y,  eta_A_z,
													eta_B_x,  eta_B_y,  eta_B_z,
													eta_OH_x, eta_OH_y, eta_OH_z,
													eta_H_x,  eta_H_y,  eta_H_z,
													eta_H2O_,
													c_H2O_,
													c_A_,
													c_B_,
													c_OH_,
													c_H_,
													F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz
													},
													{1},
													QGauss<spacedim>(degree+1),
													global_data,
													4,
													method,
													alpha);

	// part 7 of omega (related to flux of A+ in solution bath)
	OmegaDualIonDissipation00<spacedim> omega_7({c_A_, eta_A_x, eta_A_y, eta_A_z, eta_A_, c_A_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
												{1},
												QGauss<spacedim>(degree+1),
												global_data,
												D_A / (R*T),
												V_m_H2O,
												method,
												alpha);

	// part 8 of omega (related to flux of B- in solution bath)
	OmegaDualIonDissipation00<spacedim> omega_8({c_B_, eta_B_x, eta_B_y, eta_B_z, eta_B_, c_B_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
												{1},
												QGauss<spacedim>(degree+1),
												global_data,
												D_B / (R*T),
												V_m_H2O,
												method,
												alpha);

	// part 9 of omega (related to flux of H+ in solution bath)
	OmegaDualIonDissipation00<spacedim> omega_9({c_H_, eta_H_x, eta_H_y, eta_H_z, eta_H_, c_H_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
												{1},
												QGauss<spacedim>(degree+1),
												global_data,
												D_H / (R*T),
												V_m_H2O,
												method,
												alpha);

	// part 10 of omega (related to flux of OH- in solution bath)
	OmegaDualIonDissipation00<spacedim> omega_10(	{c_OH_, eta_OH_x, eta_OH_y, eta_OH_z, eta_OH_, c_OH_, c_H2O_, F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
													{1},
													QGauss<spacedim>(degree+1),
													global_data,
													D_OH / (R*T),
													V_m_H2O,
													method,
													alpha);

	// part 11 of omega (related to oxidation reaction of the electrolysis)
	UExternal<spacedim> u_external(U_th, delta_U, t);
	OmegaElectrolysis00<spacedim> omega_11(	{delta_eta_0_ox},
											{0,1},
											QGauss<spacedim-1>(degree+1),
											global_data,
											F,
											R_if_ox,
											A_e_ox,
											phi_c_ox,
											u_external,
											method,
											alpha,
											0.0);
	TotalPotentialContribution<spacedim> omega_11_tpc(omega_11);

	// part 12 of omega (related to reduction reaction of the electrolysis)
	OmegaElectrolysis00<spacedim> omega_12(	{delta_eta_0_red},
											{0,1},
											QGauss<spacedim-1>(degree+1),
											global_data,
											F,
											R_if_red,
											A_e_red,
											phi_c_red,
											u_external,
											method,
											alpha,
											0.0);
	TotalPotentialContribution<spacedim> omega_12_tpc(omega_12);


	// part 13 of omega (related to dissociation of CH in gel, used only if dissociation of CH groups is taken into account)
	OmegaMixedTerm00<spacedim> omega_13({c_C_, eta_C},
										{0},
										QGauss<spacedim>(degree+1),
										global_data,
										method,
										alpha);


	// part 14 of omega in solution bath (related to water flux), used for infinitesimal viscosity approach
	OmegaDualFluidDissipation05<spacedim> omega_14(	{
													v_s_x, v_s_y, v_s_z,
													c_H2O_,
													u_x, u_y, u_z,
													eta_H2O_x,  eta_H2O_y,  eta_H2O_z,
													eta_A_x,  eta_A_y,  eta_A_z,
													eta_B_x,  eta_B_y,  eta_B_z,
													eta_OH_x, eta_OH_y, eta_OH_z,
													eta_H_x,  eta_H_y,  eta_H_z,
													eta_H2O_,
													c_H2O_,
													c_A_,
													c_B_,
													c_OH_,
													c_H_,
													F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz
													},
													{1},
													QGauss<spacedim>(degree+1),
													global_data,
													4,
													method,
													alpha);

	// part 15 of omega in solution bath (constrain velocity field to satisfy equilibrium equations for quasi-static Newtonian fluid), used for infinitesimal viscosity approach
	OmegaLagrangeViscousDissipation00<spacedim> omega_15(	{v_s_xx, v_s_xy, v_s_xz, v_s_yx, v_s_yy, v_s_yz, v_s_zx, v_s_zy, v_s_zz,
															f_s_xx, f_s_xy, f_s_xz, f_s_yx, f_s_yy, f_s_yz, f_s_zx, f_s_zy, f_s_zz,
															F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
															{1},
															QGauss<spacedim>(degree+2),
															global_data,
															method,
															alpha);
	TotalPotentialContribution<spacedim> omega_15_tpc(omega_15);

	// part 16 of omega in solution bath (constrain velocity field to satisfy equilibrium equations for quasi-static Newtonian fluid), used for infinitesimal viscosity approach
	OmegaLagrangeIncompressibility01<spacedim> omega_16({xi_s_x, xi_s_y, xi_s_z,
														f_s_x, f_s_y, f_s_z,
														F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz},
														{1},
														QGauss<spacedim>(degree+2),
														global_data,
														method,
														alpha);
	TotalPotentialContribution<spacedim> omega_16_tpc(omega_16);

	// part 17 of omega in solution bath (constrain velocity field to satisfy equilibrium equations for quasi-static Newtonian fluid), used for infinitesimal viscosity approach
	OmegaLagrangeIncompressibility02<spacedim> omega_17({t_S_,
														f_s_x_minus, f_s_y_minus, f_s_z_minus,
														F_xx_minus, F_xy_minus, F_xz_minus, F_yx_minus, F_yy_minus, F_yz_minus, F_zx_minus, F_zy_minus, F_zz_minus},
														{0,1,3,5,6},
														QGauss<spacedim-1>(degree+2),
														global_data,
														false,
														method,
														alpha);
	TotalPotentialContribution<spacedim> omega_17_tpc(omega_17);

	// part 18 of omega in solution bath (constrain velocity field to satisfy equilibrium equations for quasi-static Newtonian fluid), used for infinitesimal viscosity approach
	// This is an empirical contribution stabilizing the formulation.
	// The underlying problem is as follows:
	// Without the action of omega_17, f_s is constrained by omega_16 to be orthogonal to the space used for grad_eta_H2O (and grad_xi_s).
	// As a consequence, the constraint imposed by omega_15 on v_s is such that the null space of this constraint contains no non-empty subspace being orthogonal to the space of grad_eta_H2O.
	// This would result in principle in a stable formulation. However, the normal component of f_s needs to be constrained through omega_17 so as to avoid distribution-like behavior for xi_s. This does, however,
	// result in a situation where the null space of the constraint omega_15 now contains a non-empty subspace of v_s being orthogonal to the space of grad_eta_H2O. As a result, the entire formulation would not be inf-sup stable.
	// In order to remedy this situation, the non-empty subspace of v_s must be eliminated. An explicit construction of this subspace appears to be non-trivial.
	// However, empirically, the contribution omega_18 effects the elimination of this subspace as evidenced by optimal rates of convergence.
	// Further theoretical investigations are necessary in this respect. In this context, it would also be desirable to allow t_S to be discontinuous at corners.
	OmegaFluidIncompressibility05<spacedim> omega_18({	v_s_xx_minus, v_s_xy_minus, v_s_xz_minus, v_s_yx_minus, v_s_yy_minus, v_s_yz_minus, v_s_zx_minus, v_s_zy_minus, v_s_zz_minus,
														eta_S_,
														F_xx_minus, F_xy_minus, F_xz_minus, F_yx_minus, F_yy_minus, F_yz_minus, F_zx_minus, F_zy_minus, F_zz_minus},
														{0,1,3,5,6},
														QGauss<spacedim-1>(degree+2),
														global_data,
														1.0,
														method,
														alpha);
	TotalPotentialContribution<spacedim> omega_18_tpc(omega_18);

	// part 19 of omega in solution bath (constrain tangential velocity field)
	DependentField<spacedim-1, spacedim> dummy_S("dummy_S");
	OmegaZeroTangentialFlux00<spacedim> omega_19({	v_s_x_minus, v_s_y_minus, v_s_z_minus,
													dummy_S, dummy_S, dummy_S,
													eta_t_S_x, eta_t_S_y, eta_t_S_z,
													F_xx_minus, F_xy_minus, F_xz_minus, F_yx_minus, F_yy_minus, F_yz_minus, F_zx_minus, F_zy_minus, F_zz_minus},
													{0,1,5},
													QGauss<spacedim-1>(degree+2),
													global_data,
													false,
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_19_tpc(omega_19);

	// part 20 of omega in solution bath (constrain tangential velocity field)
	OmegaZeroTangentialFlux00<spacedim> omega_20({	u_x_minus, u_y_minus, u_z_minus,
													v_s_x_minus, v_s_y_minus, v_s_z_minus,
													eta_t_S_x, eta_t_S_y, eta_t_S_z,
													F_xx_minus, F_xy_minus, F_xz_minus, F_yx_minus, F_yy_minus, F_yz_minus, F_zx_minus, F_zy_minus, F_zz_minus},
													{6},
													QGauss<spacedim-1>(degree+2),
													global_data,
													false,
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_20_tpc(omega_20);

	// part 21 of omega in solution bath (conjugate constraint to tangential velocity field constraint)
	OmegaZeroTangentialFlux00<spacedim> omega_21({	t_t_S_x, t_t_S_y, t_t_S_z,
													dummy_S, dummy_S, dummy_S,
													f_s_x_minus, f_s_y_minus, f_s_z_minus,
													F_xx_minus, F_xy_minus, F_xz_minus, F_yx_minus, F_yy_minus, F_yz_minus, F_zx_minus, F_zy_minus, F_zz_minus},
													{0,1,5},
													QGauss<spacedim-1>(degree+2),
													global_data,
													false,
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_21_tpc(omega_21);

	// part 22 of omega in solution bath (conjugate constraint to tangential velocity field constraint)
	OmegaZeroTangentialFlux00<spacedim> omega_22({	t_t_S_x, t_t_S_y, t_t_S_z,
													dummy_S, dummy_S, dummy_S,
													f_s_x_minus, f_s_y_minus, f_s_z_minus,
													F_xx_minus, F_xy_minus, F_xz_minus, F_yx_minus, F_yy_minus, F_yz_minus, F_zx_minus, F_zy_minus, F_zz_minus},
													{6},
													QGauss<spacedim-1>(degree+2),
													global_data,
													false,
													method,
													alpha);
	TotalPotentialContribution<spacedim> omega_22_tpc(omega_22);

	// combine scalar functional as required by local eliminiation approach
	vector<ScalarFunctional<spacedim, spacedim>*> scalar_functionals_g, scalar_functionals_s;
	ScalarFunctional<spacedim, spacedim>* omega_6_14;
	if(with_infinitesimal_viscosity)
		omega_6_14 = &omega_14;
	else
		omega_6_14 = &omega_6;
	if(with_CH_dissociation)
	{
		scalar_functionals_g = {&psi_U_1,
								&psi_U_2,
								&psi_U_3,
								&psi_U_4,
								&psi_U_5,
								&psi_U_6,
								&psi_U_7,
								&psi_U_13,
								&psi_U_14,
								&psi_P_1,
								&psi_P_3,
								&psi_P_5,
								&omega_1,
								&omega_2,
								&omega_3,
								&omega_4,
								&omega_5,
								&omega_13};

		scalar_functionals_s = { &psi_U_8,
								 &psi_U_9,
								 &psi_U_10,
								 &psi_U_11,
								 &psi_U_12,
								 &psi_P_2,
								 &psi_P_4,
								 &psi_P_6,
								 omega_6_14,
								 &omega_7,
								 &omega_8,
								 &omega_9,
								 &omega_10};
	}
	else
	{
		scalar_functionals_g = {&psi_U_1,
								&psi_U_2,
								&psi_U_3,
								&psi_U_4,
								&psi_U_5,
								&psi_U_6,
								&psi_U_7,
								&psi_P_1,
								&psi_P_3,
								&psi_P_5,
								&omega_1,
								&omega_2,
								&omega_3,
								&omega_4,
								&omega_5};

		scalar_functionals_s = { &psi_U_8,
								 &psi_U_9,
								 &psi_U_10,
								 &psi_U_11,
								 &psi_U_12,
								 &psi_P_2,
								 &psi_P_4,
								 &psi_P_6,
								 omega_6_14,
								 &omega_7,
								 &omega_8,
								 &omega_9,
								 &omega_10};
	}

	ScalarFunctionalLocalElimination<spacedim, spacedim> psi_g(scalar_functionals_g, "psi_g");
	TotalPotentialContribution<spacedim> psi_g_tpc(psi_g);

	ScalarFunctionalLocalElimination<spacedim, spacedim> psi_s(scalar_functionals_s, "psi_s");
	TotalPotentialContribution<spacedim> psi_s_tpc(psi_s);

	// finally assemble incremental potential as sum of individual contributions defined earlier
	TotalPotential<spacedim> total_potential;
	total_potential.add_total_potential_contribution(psi_g_tpc);
	total_potential.add_total_potential_contribution(psi_s_tpc);
	total_potential.add_total_potential_contribution(omega_11_tpc);
	total_potential.add_total_potential_contribution(omega_12_tpc);
	if(with_infinitesimal_viscosity)
	{
		total_potential.add_total_potential_contribution(omega_15_tpc);
		total_potential.add_total_potential_contribution(omega_16_tpc);
		total_potential.add_total_potential_contribution(omega_17_tpc);
		total_potential.add_total_potential_contribution(omega_18_tpc);
		total_potential.add_total_potential_contribution(omega_19_tpc);
		total_potential.add_total_potential_contribution(omega_20_tpc);
		total_potential.add_total_potential_contribution(omega_21_tpc);
		total_potential.add_total_potential_contribution(omega_22_tpc);
	}

/***************************
 * Solution of the problem *
 ***************************/

	BlockSolverWrapperMUMPS solver_wrapper_mumps;
	solver_wrapper_mumps.icntl[2] = 6;		// standard output stream
	solver_wrapper_mumps.icntl[3] = 1;		// print level
	solver_wrapper_mumps.icntl[7] = 0;		// row scaling
	solver_wrapper_mumps.icntl[6] = 5;		// use METIS ordering
	solver_wrapper_mumps.icntl[9] = 10;		// maximum number of iterative refinements
	solver_wrapper_mumps.icntl[27] = 1;		// sequential calculation
	solver_wrapper_mumps.icntl[13] = 200;	// space for pivoting
	solver_wrapper_mumps.cntl[0] = 0.1;		// pivoting threshold
	solver_wrapper_mumps.analyze = 1;		// only analyze matrix structure in first step as it does not change subsequently

	// set up finite element model
	FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper_mumps, true, true);

	// incorporate trial step
	TrialStep<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> trial_step(fe_model, {&omega_11, &omega_12}, phi_e);

	// string for file names
	const string variant_string = "_a=" + Utilities::to_string(alpha)
								+ "_met=" + Utilities::to_string(method)
								+ "_p=" + Utilities::to_string(degree)
								+ "_m_t=" + Utilities::to_string(m_t)
								+ "_m_h=" + Utilities::to_string(m_h);

	// define postprocessors
	PostprocessorConcentration<spacedim> pp_A("c_A_", fe_model.get_assembly_helper().get_u_omega_global_component_index(c_A), fe_model.get_assembly_helper().get_u_omega_global_component_index(u));
	fe_model.attach_data_postprocessor_domain(pp_A);
	PostprocessorConcentration<spacedim> pp_B("c_B_", fe_model.get_assembly_helper().get_u_omega_global_component_index(c_B), fe_model.get_assembly_helper().get_u_omega_global_component_index(u));
	fe_model.attach_data_postprocessor_domain(pp_B);
	PostprocessorConcentration<spacedim> pp_H("c_H_", fe_model.get_assembly_helper().get_u_omega_global_component_index(c_H), fe_model.get_assembly_helper().get_u_omega_global_component_index(u));
	fe_model.attach_data_postprocessor_domain(pp_H);
	PostprocessorConcentration<spacedim> pp_OH("c_OH_", fe_model.get_assembly_helper().get_u_omega_global_component_index(c_OH), fe_model.get_assembly_helper().get_u_omega_global_component_index(u));
	fe_model.attach_data_postprocessor_domain(pp_OH);
	PostprocessorPH<spacedim> pp_pH("pH", fe_model.get_assembly_helper().get_u_omega_global_component_index(c_H), fe_model.get_assembly_helper().get_u_omega_global_component_index(u), fe_model.get_assembly_helper().get_u_omega_global_component_index(c_H2O), c_H_g_ref, c_H_s_ref, c_H2O_g_ref, c_H2O_s_ref, n_p_ref, c_std, eps);
	fe_model.attach_data_postprocessor_domain(pp_pH);

	// the actual computation loop
	bool error = false;
	const auto time_instants = get_time_instants(0.0, t, N, q, m_t);
	for(unsigned int step = 1; step < time_instants.size(); ++step)
	{
		cout << "time step " << step <<" of " << time_instants.size() - 1 << endl;
		cout << "New time =  " << time_instants[step] << endl;

		const double new_time = time_instants[step];
		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			if(write_output)
				fe_model.write_output_independent_fields("results/output_files/domain" + variant_string, "results/output_files/interface" + variant_string, cell_divisions);
		}
		else
		{
			cout << "ERROR, Computation failed!" << endl;
			error = true;
			global_data.print_error_messages();
			break;
		}
		cout << endl;
	}
	if(!write_output)
		fe_model.write_output_independent_fields("results/output_files/domain" + variant_string, "results/output_files/interface" + variant_string, cell_divisions);
	global_data.print_error_messages();

	// write reference solution if requested
	if(write_reference)
	{
		fe_model.write_solution_to_file(result_file);
		return {1.0 / (double)N, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	}
	// compare with reference solution if requested
	else
	{
		double d_linfty = 1e16;
		double d_l2 = 1e16;
		double d_linfty_grad_u = 1e16;
		double d_linfty_c_H2O = 1e16;
		double d_linfty_c_A = 1e16;
		double d_linfty_c_B = 1e16;
		double d_linfty_c_H = 1e16;
		double d_linfty_c_OH = 1e16;
		double d_linfty_c_C = with_CH_dissociation ? 1e16 : 0.0;
		double d_linfty_grad_l_s = 1e16;
		double d_l2_grad_u = 1e16;
		double d_l2_c_H2O = 1e16;
		double d_l2_c_A = 1e16;
		double d_l2_c_B = 1e16;
		double d_l2_c_H = 1e16;
		double d_l2_c_OH = 1e16;
		double d_l2_c_C =  with_CH_dissociation ? 1e16 : 0.0;
		double d_l2_grad_l_s = 1e16;

		if(!error)
		{
			FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model_reference(total_potential, tria_system_ref, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper_mumps, true, true);
			fe_model_reference.read_solution_from_file(result_file);

			ComponentMask cm_domain(fe_model.get_assembly_helper().get_dof_handler_system().get_dof_handler_domain().get_fe_collection().n_components(), false);

			// grad_u
			for(unsigned int i = 0; i < spacedim; ++i)
				cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(u)+i, true);
			d_linfty_grad_u = fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::W1infty_seminorm, cm_domain, ComponentMask(), 0.0).first;
			d_l2_grad_u = fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::H1_seminorm, cm_domain, ComponentMask(), 0.0).first;
			for(unsigned int i = 0; i < spacedim; ++i)
				cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(u)+i, false);

			// c_H2O
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_H2O), true);
			d_linfty_c_H2O = 1.0 / c_H2O_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::Linfty_norm, cm_domain, ComponentMask(), 0.0).first;
			d_l2_c_H2O = 1.0 / c_H2O_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::L2_norm, cm_domain, ComponentMask(), 0.0).first;
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_H2O), false);

			// c_A
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_A), true);
			d_linfty_c_A = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::Linfty_norm, cm_domain, ComponentMask(), 0.0).first;
			d_l2_c_A = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::L2_norm, cm_domain, ComponentMask(), 0.0).first;
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_A), false);

			// c_B
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_B), true);
			d_linfty_c_B = 1.0 / c_B_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::Linfty_norm, cm_domain, ComponentMask(), 0.0).first;
			d_l2_c_B = 1.0 / c_B_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::L2_norm, cm_domain, ComponentMask(), 0.0).first;
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_B), false);

			// c_H
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_H), true);
			d_linfty_c_H = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::Linfty_norm, cm_domain, ComponentMask(), 0.0).first;
			d_l2_c_H = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::L2_norm, cm_domain, ComponentMask(), 0.0).first;
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_H), false);

			// c_OH
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_OH), true);
			d_linfty_c_OH = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::Linfty_norm, cm_domain, ComponentMask(), 0.0).first;
			d_l2_c_OH = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::L2_norm, cm_domain, ComponentMask(), 0.0).first;
			cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_OH), false);

			// c_C
			if(with_CH_dissociation)
			{
				cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_C), true);
				d_linfty_c_C = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::Linfty_norm, cm_domain, ComponentMask(), 0.0).first;
				d_l2_c_C = 1.0 / c_A_s_ref * fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::L2_norm, cm_domain, ComponentMask(), 0.0).first;
				cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(c_C), false);
			}

			// grad_l_s
			for(unsigned int i = 0; i < spacedim; ++i)
				cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(l_s)+i, true);
			d_linfty_grad_l_s = fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::W1infty_seminorm, cm_domain, ComponentMask(), 0.0).first;
			d_l2_grad_l_s = fe_model_reference.compute_distance_to_other_solution(fe_model, QGauss<spacedim>(degree+1), QGauss<spacedim-1>(degree+1), VectorTools::NormType::H1_seminorm, cm_domain, ComponentMask(), 0.0).first;
			for(unsigned int i = 0; i < spacedim; ++i)
				cm_domain.set(fe_model_reference.get_assembly_helper().get_u_omega_global_component_index(l_s)+i, false);

			// combined errors
			d_linfty = max(max(max(max(max(max(d_linfty_grad_u, d_linfty_c_H2O), d_linfty_c_A), d_linfty_c_H), d_linfty_c_OH), d_linfty_c_C), d_linfty_c_B);
			d_l2 = sqrt(d_l2_grad_u * d_l2_grad_u + d_l2_c_H2O * d_l2_c_H2O + d_l2_c_A * d_l2_c_A + d_l2_c_B * d_l2_c_B + d_l2_c_H * d_l2_c_H + d_l2_c_OH * d_l2_c_OH + d_l2_c_C * d_l2_c_C);
		}

		return {1.0 / (double)N, d_linfty_grad_u, d_linfty_c_H2O, d_linfty_c_A, d_linfty_c_B, d_linfty_c_H, d_linfty_c_OH, d_linfty_grad_l_s, d_l2_grad_u, d_l2_c_H2O, d_l2_c_A, d_l2_c_B, d_l2_c_H, d_l2_c_OH, d_l2_grad_l_s, d_linfty, d_l2};
	}

}


int main(int argc, char **argv)
{

	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	const unsigned int m_t_max = 0;	// maximum number of refinements in time for convergence study
	const unsigned int m_t = 1;		// number of refinements in time to be used for convergence study in space

	// polynomial degrees of finite elements to be studied, together with maxmimum number of refinements in space to be used for spatial convergence study and number of refinements in space to be used for
	// temporal convergence study
	vector<tuple<unsigned int, unsigned int, unsigned int>> degrees_m_h_max_m_h;
	degrees_m_h_max_m_h.push_back(make_tuple(1, 4, 0));
	degrees_m_h_max_m_h.push_back(make_tuple(2, 3, 0));

	// time integration methods to be studied
	vector<pair<double, unsigned int>> methods_t;
	methods_t.push_back(make_pair(0.5, 1));
	methods_t.push_back(make_pair(1.0, 0));
	methods_t.push_back(make_pair(0.5, 2));

	for(const auto degree_m_h_max_m_h : degrees_m_h_max_m_h)
	{
		for(const auto method : methods_t)
		{

// convergence study in time
			const string variant_string_t = "_a=" + Utilities::to_string(method.first)
										  + "_met=" + Utilities::to_string(method.second)
										  + "_p=" + Utilities::to_string(get<0>(degree_m_h_max_m_h))
										  + "_t";


			const string file_name_res_t	= "results/results" + variant_string_t + ".dat";				// file where results are stored
			const string file_name_ref_t	= "results/results" + variant_string_t + "_ref.dat";			// file where reference solution is stored

			// generate reference solution
			auto result_data_ref = solve(m_t_max, get<2>(degree_m_h_max_m_h), method.first, method.second, get<0>(degree_m_h_max_m_h), file_name_ref_t, true, get<2>(degree_m_h_max_m_h), false);

			// clear file
			FILE* printout_t = fopen(file_name_res_t.c_str(),"w");
			fclose(printout_t);

			// compare
			for(unsigned int m = 0; m < m_t_max; ++m)
			{
				const auto result_data = solve(m, get<2>(degree_m_h_max_m_h), method.first, method.second, get<0>(degree_m_h_max_m_h), file_name_ref_t, false, get<2>(degree_m_h_max_m_h), false);
				FILE* printout_t_ = fopen(file_name_res_t.c_str(),"a");
				fprintf(printout_t_, "%- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e\n", 1.0/pow(2.0, (double)m), result_data[1], result_data[2], result_data[3], result_data[4], result_data[5], result_data[6], result_data[7], result_data[8], result_data[9], result_data[10], result_data[11], result_data[12], result_data[13], result_data[14], result_data[15], result_data[16]);
				fclose(printout_t_);
			}

// convergence study in space
			const string variant_string_h = "_a=" + Utilities::to_string(method.first)
										  + "_met=" + Utilities::to_string(method.second)
										  + "_p=" + Utilities::to_string(get<0>(degree_m_h_max_m_h))
										  + "_h";

			const string file_name_res_h	= "results/results" + variant_string_h + ".dat";				// file where results are stored
			const string file_name_ref_h	= "results/results" + variant_string_h + "_ref.dat";			// file where reference solution is stored

			// generate reference solution
			result_data_ref = solve(m_t, get<1>(degree_m_h_max_m_h), method.first, method.second, get<0>(degree_m_h_max_m_h), file_name_ref_h, true, get<1>(degree_m_h_max_m_h), false);

			// clear file
			FILE* printout_h = fopen(file_name_res_h.c_str(),"w");
			fclose(printout_h);

			// compare
			for(unsigned int m = 0; m < get<1>(degree_m_h_max_m_h); ++m)
			{
				const auto result_data = solve(m_t, m, method.first, method.second, get<0>(degree_m_h_max_m_h), file_name_ref_h, false, get<1>(degree_m_h_max_m_h), false);
				FILE* printout_h_ = fopen(file_name_res_h.c_str(),"a");
				fprintf(printout_h_, "%- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e %- 1.4e\n", 1.0/pow(2.0, (double)m), result_data[1], result_data[2], result_data[3], result_data[4], result_data[5], result_data[6], result_data[7], result_data[8], result_data[9], result_data[10], result_data[11], result_data[12], result_data[13], result_data[14], result_data[15], result_data[16]);
				fclose(printout_h_);
			}

		}
	}
}
