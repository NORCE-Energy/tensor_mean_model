# Using the Egg model (oil-water compressible)
# Jansen, Jan-Dirk, et al. "The egg model–a geological ensemble for reservoir
# simulation." Geoscience Data Journal 1.2 (2014): 192-195.](https://doi.org/10.1002/gdj3.21)
# With help from Olav Moyner (SINTEF)

# REMOVE two producers (PROD3 and PROD4)

using Jutul, JutulDarcy, DelimitedFiles, HYPRE, Statistics 
using GLMakie
#import Pkg
#Pkg.add("Optimization")
using Optimization, OptimizationPRIMA


# The reference model has vertical producers at PROD1: (16,43), PROD2: (35,40), PROD3: (23,16), PROD4: (43,18) 
# Convert to physical coordinates
vert_ref = [16,43, 35,40, 23,16, 43,18] *8.0 .- 4.0

vert_ref = [39.00,  378.70,  39.19,  328.16,  409.65,  37.73,  409.74, 103.39]

# Read input for base case from an Eclipse data file
egg_dir = JutulDarcy.GeoEnergyIO.test_input_file_path("EGG")
data_path = joinpath(egg_dir, "EGG_15yr.DATA")
data = JutulDarcy.GeoEnergyIO.parse_data_file(data_path)
size_perm = size(data["GRID"]["PERMX"])
length_perm = size_perm[1]*size_perm[2]*size_perm[3]  # the three dimensions of the grid
Ne=100

Darcy, milli = si_units(:darcy, :milli)
mD = milli*Darcy
day = si_unit(:day)
conv_rate =  si_unit(:day)/si_unit(:stb)

case = setup_case_from_data_file(data)
(; model, state0, forces, parameters, dt) = case

reservoir = reservoir_domain(model)

# Now create a new model with modified permeability (the mean model)
# Compute the variance of the revised  log permeability field for the anisotropy computation
var_log_perm = 4. * var(log.(data["GRID"]["PERMX"]))

# Compute the "mean model" for optimization from the ensemble of realizations
# First compute the isotropic mean permeability model
sum_logperm = zeros(length_perm)
for i = 1:Ne
    perm_filename = string("PERM",string(i),"_ECL.INC")
    perm_data_path = joinpath(egg_dir, perm_filename)
    permR_all = readdlm(perm_data_path)
    permi_flat = vec(transpose(permR_all[2:4201,1:6]))
    #rev_permi_flat = 2. * log10.(permi_flat) .- 3.5 
    #permi = reshape(permi_flat, size_perm)
    sum_logperm = sum_logperm + (2. * log10.(permi_flat) .- 3.5)
end
mean_logperm = sum_logperm/Ne
mean_perm_flat = 10 .^ (mean_logperm)
mean_perm = reshape(mean_perm_flat, size_perm)
# end compute isotropic mean permeability model

# EFFECTIVE REL PERM CURVES FOR SW_STAR = 0.54  (SWOF_EFF)   
SWOF_EFF = [[0.1 0.0 0.7999989670634583 0.0; 
    0.12 0.0 0.7999989670634583 0.0; 
    0.14 0.0 0.7999989670634583 0.0; 
    0.16 0.0 0.7999989670634583 0.0; 
    0.18 0.0 0.7999989670634583 0.0; 
    0.2 1.2566464583257323e-48 0.7997079999999996 0.0; 
    0.22 0.0006449482294041444 0.520089765503178 0.0; 
    0.24 0.005159585835233149 0.3224687108896484 0.0; 
    0.26 0.01741360219391189 0.1885334965464025 0.0; 
    0.28 0.041276686681865245 0.10098568678016476 0.0; 
    0.3 0.0806185286755179 0.04759883343877225 0.0; 
    0.32 0.09284496322701462 0.038170167192487434 0.0; 
    0.34 0.10142023675277125 0.03274742479766238 0.0; 
    0.36 0.1105079084650003 0.028036694209671933 0.0; 
    0.38 0.12012285301153361 0.023929131700914444 0.0; 
    0.4 0.13027995750265045 0.02033981709165994 0.0; 
    0.42 0.14099413269407982 0.017201823246512232 0.0; 
    0.44 0.15228032681235026 0.014461403995413569 0.0; 
    0.46 0.16415354096190005 0.012074279174783958 0.0; 
    0.48 0.17662884514291596 0.01000288872735039 0.0; 
    0.5 0.1897213940881197 0.008214454462999454 0.0; 
    0.52 0.20344644232371198 0.00667968225634627 0.0; 
    0.54 0.21781935804082203 0.005371939372875167 0.0; 
    0.56 0.23285563551598987 0.004266753169045456 0.0; 
    0.58 0.24857090593984546 0.003341505084571983 0.0; 
    0.6 0.2649809466045645 0.002575232765529567 0.0; 
    0.62 0.2821016884671777 0.0019484901151314848 0.0; 
    0.64 0.29994922215209535 0.0014432410187162348 0.0; 
    0.66 0.3185398024866817 0.001042776774570764 0.0; 
    0.68 0.3378898516821665 0.0007316535741111463 0.0; 
    0.7 0.35801596128173435 0.000495648511205371 0.0; 
    0.72 0.378934893000733 0.00032173298271154227 0.0; 
    0.74 0.40066357858255675 0.0001980621739883226 0.0; 
    0.76 0.4232191187892849 0.00011397909522953276 0.0; 
    0.78 0.4466187816397388 6.0031500550283834e-5 0.0; 
    0.8 0.4708799999999997 2.7999999999694936e-5 0.0; 
    0.82 0.49602036862324983 1.0935739594708505e-5 0.0; 
    0.84 0.5220576407273964 3.206144022493973e-6 0.0; 
    0.86 0.5490097241906676 5.423321314257379e-7 0.0; 
    0.88 0.5768946774373372 2.0477469164781326e-8 0.0; 
    0.9 0.6057307050781585 0.0 0.0]]; 


# Computing the anisotropy for the mean model from Ababou (1996)
length_scales = [3.0, 12.0, 12.0]

l_h= 1. / (sum(1 ./ length_scales)/length(length_scales))
p = ones(3)
for i = 1:3
    p[i] = exp( 0.5*var_log_perm*( 1. - (2. * l_h) / (3. * length_scales[i]) ) )
end
println(p)
# These are the multipliers for each direction to create anisotropy
# load the anisotropic permeability field into the model 
data["GRID"]["PERMX"] .= p[1]*mean_perm*mD
data["GRID"]["PERMY"] .= p[2]*mean_perm*mD
data["GRID"]["PERMZ"] .= p[3]*mean_perm*mD
data["PROPS"]["SWOF"] .= SWOF_EFF 
case = setup_case_from_data_file(data)
(; model, state0, forces, parameters, dt) = case

# This is now using the mean permeability field
reservoir = reservoir_domain(model)

# ... and now specify the well trajectories of the vertical reference producers
# PROD1
traj1 = [
    vert_ref[1] vert_ref[2] 4002.;    # heel
    vert_ref[3] vert_ref[4] 4022.     # toe
]
P1 = setup_well_from_trajectory(reservoir, traj1, name = :PROD1)
traj2 = [
    vert_ref[5] vert_ref[6] 4002.
    vert_ref[7] vert_ref[8] 4022.
]
P2 = setup_well_from_trajectory(reservoir, traj2, name = :PROD2)
I4 = setup_well(reservoir, (27,29), name = :INJECT4)

new_model, new_parameters = setup_reservoir_model(reservoir, model, wells = [I4,P1,P2]);

#fig = plot_reservoir(new_model, title = "New model", alpha = 0.0, edge_color = :black)
#lines!(fig.current_axis[], traj', color = :red)
#fig

new_state0 = setup_reservoir_state(new_model, state0)
new_control = Dict()
new_limits = Dict()

facility_forces = forces[1][:Facility]

ictrl = facility_forces.control[:INJECT4]
ilims = facility_forces.limits[:INJECT4]

pctrl = facility_forces.control[:PROD1]
plims = facility_forces.limits[:PROD1]

new_control[:INJECT4] = ictrl
new_control[:PROD1] = pctrl
new_control[:PROD2] = pctrl

new_limits[:INJECT4] = ilims
new_limits[:PROD1] = plims
new_limits[:PROD2] = plims

new_forces = setup_reservoir_forces(new_model, control = new_control, limits = new_limits)


#simulate_reservoir(new_state0, new_model, dt, forces = new_forces, parameters = new_parameters)



function npv_fun(xy,p)
"""
The xy is the vector of heel and toe x-y locations for the 4 producers.
The depth of the heel is always in the center of the grid on top layer (4002 m).
The depth of the toe is always in the center of the grid on bottom layer (4022 m).
"""
    traj1 = [
        xy[1] xy[2] 4002.;    # heel
        xy[3] xy[4] 4022.     # toe
    ]
    P1 = setup_well_from_trajectory(reservoir, traj1, name = :PROD1)
    traj2 = [
        xy[5] xy[6] 4002.
        xy[7] xy[8] 4022.
    ]
    P2 = setup_well_from_trajectory(reservoir, traj2, name = :PROD2)
    I4 = setup_well(reservoir, (27,29), name = :INJECT4)

    new_model, new_parameters = setup_reservoir_model(reservoir, model, wells = [I4,P1,P2]);
    new_state0 = setup_reservoir_state(new_model, state0)
    new_control = Dict()
    new_limits = Dict()
    
    facility_forces = forces[1][:Facility]
    
    ictrl = facility_forces.control[:INJECT4]
    ilims = facility_forces.limits[:INJECT4]
    
    pctrl = facility_forces.control[:PROD1]
    plims = facility_forces.limits[:PROD1]

    new_control[:INJECT4] = ictrl
    new_control[:PROD1] = pctrl
    new_control[:PROD2] = pctrl

    new_limits[:INJECT4] = ilims
    new_limits[:PROD1] = plims
    new_limits[:PROD2] = plims
    
    new_forces = setup_reservoir_forces(new_model, control = new_control, limits = new_limits)
    
    wd, states, t = simulate_reservoir(new_state0, new_model, dt, forces = new_forces, parameters = new_parameters)

    wells = deepcopy(wd.wells)
    global time_jutul = deepcopy(wd.time)
    N = length(time_jutul)
    # Compute field rates from a simulation run 
    inj = Symbol[]
    prod = Symbol[]
    qf_oil = zeros(N)    # field oil production rate
    qf_wtr_prod = zeros(N)    # field water production rate
    qf_wtr_inj = zeros(N)     # field water injection rate
    for (wellname, well) in pairs(wells)
        qts = well[:wrat] + well[:orat]
        if sum(qts) > 0
            push!(inj, wellname)
            qf_wtr_inj = qf_wtr_inj + well[:wrat]
        else
            push!(prod, wellname)
            qf_oil = qf_oil - well[:orat]
            qf_wtr_prod = qf_wtr_prod - well[:wrat]
        end
    end
    global npv = NPV(qf_oil, qf_wtr_prod, qf_wtr_inj)
    println("npv = ", npv, "   loc PROD1: (", xy[1:4],  "   loc PROD2: (", xy[5:8])
    #println("          loc PROD3: (", xy[9:10],  "   loc PROD4: (",xy[13:14])
    push!(npv_all, copy(npv))
    push!(xy_iter_all, copy(xy) )
    return -npv
end

function NPV(oil_rates, water_rates_prd, water_rates_inj)
    # Assumes that rates are in SI units
    drate = 0.08   # annual discount rate 
    price_oil = 85.   # dollars per barrel
    cost_water_prod = 5.  # dollars per barrel
    cost_water_inj = 15.  # dollars per barrel
    conv_rate =  si_unit(:day)/si_unit(:stb)    # convert rates from SI to BPD
     
    N = length(oil_rates)
    mean_oil_rates = (oil_rates[2:N] + oil_rates[2:N]) ./ 2.
    mean_oil_rates_all = pushfirst!(mean_oil_rates, oil_rates[1]) .* conv_rate   # convert oil rate from SI to BPD
    mean_wtr_rates_prd = (water_rates_prd[2:N] + water_rates_prd[2:N]) ./ 2.
    mean_wtr_rates_prd_all = pushfirst!(mean_wtr_rates_prd, water_rates_prd[1]) .* conv_rate   # convert oil rate from SI to BPD
    mean_wtr_rates_inj = (water_rates_inj[2:N] + water_rates_inj[2:N]) ./ 2.
    mean_wtr_rates_inj_all = pushfirst!(mean_wtr_rates_inj, water_rates_inj[1]) .* conv_rate   # convert oil rate from SI to BPD

    ttest = time_jutul./day     # where does it get this?
    ave_t = (ttest[2:N] + ttest[1:N-1]) ./ 2.
    ave_t_all = pushfirst!(ave_t, ttest[1])
    dttest = (ttest[2:N] - ttest[1:N-1])
    dttest_all = pushfirst!(dttest, ttest[1])
    discount = (1 + drate) .^ (-ave_t_all ./ 365.)
    cash_flow_discounted = ( price_oil .* mean_oil_rates_all 
        - cost_water_prod .* mean_wtr_rates_prd_all 
        - cost_water_inj .* mean_wtr_rates_inj_all )  .* dttest_all .* discount
    npv = sum(cash_flow_discounted)  
    
    return npv
end

# x,y coordinates in physical units
xy_lb = [36., 36., 36., 36., 36., 36., 36., 36.]
xy_ub = [412., 436., 412, 436., 412., 436., 412, 436.]

# Start location 1
#xy_0 = [ 135.5037,  242.0116, 135.5037,  242.0116,  306.1930, 210.1727,  306.1930, 210.1727 ]
# Start loc 3
#xy_0 = [116.0, 236.0, 340.0, 236.0];
# Start vert loc 1
#xy_0 = [132.0, 348.0, 292.0, 132.0]; 
xy_0 = [132.0, 348.0, 132.0, 348.0, 292.0, 132.0, 292.0, 132.0];   # for lateral wells


# Optimization of well locations using BOBYQA
npv_all = []
xy_iter_all = []
obj = OptimizationFunction(npv_fun)
prob = Optimization.OptimizationProblem(obj, xy_0, lb = xy_lb, ub = xy_ub)
sol = solve(prob, BOBYQA(), npt=33, maxiters = 400, rhobeg=40, rhoend=4 )
npv_millions_all = round.(npv_all ./ 10.0^4)/100.

npv_fun(sol.u, p)


output_file = open("save_opt_15yr_1inj_lateral_tensorK_effKr_startLoc1.jl","w")
write(output_file, "# generated from Lateral_tensorMM_effKr.jl  \n \n") 
write(output_file, "xy_0 = ")
show(output_file, xy_0) # writes the starting locations
write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
write(output_file, "xy_sol = ")
show(output_file, sol.u) # writes the final well locations
write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
write(output_file, "npv_millions_all = ")
show(output_file, npv_millions_all) # writes all evaluations of NPV from optimization
write(output_file, "; \n \n") # puts a semicolon to suppress the output and two line breaks
close(output_file)

maximum(npv_millions_all)

xy_sol = sol.u

# Plot the solution

# PROD1
traj1 = [
    xy_sol[1] xy_sol[2] 4002.;    # heel
    xy_sol[3] xy_sol[4] 4022.     # toe
]
P1 = setup_well_from_trajectory(reservoir, traj1, name = :PROD1)
traj2 = [
    xy_sol[5] xy_sol[6] 4002.
    xy_sol[7] xy_sol[8] 4022.
]
P2 = setup_well_from_trajectory(reservoir, traj2, name = :PROD2)
I4 = setup_well(reservoir, (27,29), name = :INJECT4)

new_model, new_parameters = setup_reservoir_model(reservoir, model, wells = [I4,P1,P2]);

fig = plot_reservoir(new_model, title = "New model", alpha = 0.0, edge_color = :black)
#lines!(fig.current_axis[], traj', color = :red)
fig


f = Figure(fontsize = 24)
ax = Axis(f[1, 1],
    title = "Vertical well opt, tensor MM and eff Kr",
    xlabel = "Function evaluation",
    ylabel = L"NPV ($10^6$ USD)",
)
#scatter(npv_all)
scatter!(
    ax,
    npv_all/1000000.,
    color = :blue,
)
vlines!(65.5, color= :red)
f
#save("NPV_iter_tensor_mean_Case5.png", f)





