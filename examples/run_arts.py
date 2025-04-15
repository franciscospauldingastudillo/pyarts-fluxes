# The proper usage is: python3 -u run_arts.py > test.log 2>&1 &

import os
import numpy as np
import FluxSimulator as fsm
from pyarts.arts import convert
from pyarts import xml
import shutil

# =============================================================================
# Create a bespoke thermodynamic profile using parameters class and ssm1d
# =============================================================================
import parameters
import ssm1d
# Step 0: Define the case and the resolution
# specify what gases and CIA-pairs to include (see generate_case)
gases     = ['H2O']
ciapairs  = []

# dynamically create an argument dictionary
def generate_args(exp, gases, ciapairs, **thermo):
    return {'exp': exp, **{f'gas{i+1}': gas for i, gas in enumerate(gases)}, 'valid_ciapairs': ciapairs, **thermo}
args = generate_args('earth',gases,ciapairs,RHs=0.75,RHmid=0.54,RHtrp=0.75,uniform=1,Ts=300,Tmid=250,Ttrp=200)
        
# create a class instance and generate an RFM case from argument dictionary (300K = 7.1 K/km; 315K = 5.0 K/km)
par = parameters.parameters(Gamma=6.5e-3)
par.generate_case(**args)

# vertical resolution
RFM      = np.arange(par.z[0],par.z[-1],1e2)

# default dataset
# z ~ m, p ~ Pa, T ~ K, Gamma ~ K/m, RH~unitless, hr~W/m3, srhr~cm*W/m3
dataset  = ({'RFM':{}}) 

# Step 1: Generate custom atmospheric profiles (p,T,z,x) at RFM and RFMi resolution
dat1 = ssm1d.atm.get_custom_atm(par, RFM)
keys = ['p', 'T', 'RH', 'rho', 'Gamma', 'cp', 'cpmol','z']
for key in keys:
    dataset['RFM'][key] = dat1[key]

# dynamically add molar mixing ratios to the dataset (signals to rfmtools how to build the RFM experiments)
for gas in gases:
    xgas_key = f'x{gas}'  # e.g., xN2, xCH4, xH2
    dataset['RFM'][xgas_key] = dat1[xgas_key]
    
# =============================================================================
# Save profiles as an atm_{}.xml file
# =============================================================================

setup_name = f"{par.case}"
data_path = f"../atmdata/{setup_name}/"

# generate the directory if it doesn't exist
os.makedirs(data_path, exist_ok=True)

atm_field = fsm.generate_gridded_field_from_profiles(
    dataset['RFM']['p'], dataset['RFM']['T'], z_field=dataset['RFM']['z'], gases={f"{gases[0]}":dataset['RFM'][f'x{gases[0]}']}, particulates={}
)

atm_field.savexml(f'{data_path}/atm_{par.case}.xml','ascii',True)
shutil.copy(f"../atmdata/single_atmosphere/aux_earth.xml", f"{data_path}/aux_{par.case}.xml")


# =============================================================================
# Paths and constants
# =============================================================================

#atm_filename = "atm_idx9108_lat10.975_lon36.975.xml"
#aux_filename = "aux_idx9108_lat10.975_lon36.975.xml"
# atm_filename = "atm_idx9919_lat20.975_lon-141.025.xml"
# aux_filename = "aux_idx9919_lat20.975_lon-141.025.xml"
# atm_filename = "atm_mean.xml"
# aux_filename = "aux_mean.xml"
#atm_filename = "atm_earth.xml"
#aux_filename = "aux_earth.xml"

atm_filename = f"atm_{par.case}.xml"
aux_filename = f"aux_{par.case}.xml"

# =============================================================================
# Set frequency grids
# =============================================================================


min_wvn_sw = 2000.   # [cm^-1]
max_wvn_sw = 3.3333E4  # [cm^-1]
n_freq_sw = int((max_wvn_sw-min_wvn_sw)) # using 1 cm-1 resolution in SW
wvn_sw = np.linspace(min_wvn_sw, max_wvn_sw, n_freq_sw)
f_grid_sw = convert.kaycm2freq(wvn_sw)

#min_wavelength_sw = 3e-7  # [m] or 3.33E4 cm-1
#max_wavelength_sw = 5e-6  # [m] or 2000 cm-1
#n_freq_sw = 200
#n_freq_sw  = 31333 # using 1 cm-1 resolution in SW

#wvl = np.linspace(min_wavelength_sw, max_wavelength_sw, n_freq_sw)  # [m]
#f_grid_sw = convert.wavelen2freq(wvl[::-1])  # [Hz]

min_wvn = 10  # [cm^-1]
max_wvn = 3210  # [cm^-1]
#max_wvn = 1500  # [cm^-1]
#n_freq_lw = 200
n_freq_lw = int((max_wvn-min_wvn)/par.dnu) # using 0.1 cm-1 resolution in LW
wvn = np.linspace(min_wvn, max_wvn, n_freq_lw)
f_grid_lw = convert.kaycm2freq(wvn)


# =============================================================================
# load data and prepare input
# =============================================================================

# load atmosphere
atm = xml.load(os.path.join(data_path, atm_filename))

# load surface data
aux = xml.load(os.path.join(data_path, aux_filename))

# lat/lon
lat = aux[4]
lon = aux[5]

# surface altitude
surface_altitude = aux[1]  # [m]

# surface temperature
surface_temperature = aux[0]  # [K]

# surface reflectivity
surface_reflectivity_sw = 0.3
surface_reflectivity_lw = 0.05

# sun position
sun_pos = [1.495978707e11, 0.0, 36.0]  # [m], [deg], [deg]

# =============================================================================
# longwave simulation
# =============================================================================

if 0:

    LW_flux_simulator = fsm.FluxSimulator(setup_name + "_LW")
    LW_flux_simulator.ws.f_grid = f_grid_lw

    results_lw = LW_flux_simulator.flux_simulator_single_profile(
        atm,
        surface_temperature,
        surface_altitude,
        surface_reflectivity_lw,
        geographical_position=[lat, lon],
    )


# =============================================================================
# shortwave simulation
# =============================================================================
if 1:

    SW_flux_simulator = fsm.FluxSimulator(setup_name + "_SW")
    SW_flux_simulator.ws.f_grid = f_grid_sw
    SW_flux_simulator.emission = 0
    SW_flux_simulator.gas_scattering = True
    SW_flux_simulator.set_sun(sun_pos)

    results_sw = SW_flux_simulator.flux_simulator_single_profile(
        atm,
        surface_temperature,
        surface_altitude,
        surface_reflectivity_sw,
        geographical_position=[lat, lon],
    )

# =============================================================================
# plot the result
# =============================================================================

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
hrtot = results_lw['heating_rate_clearsky']+results_sw['heating_rate_clearsky']
ax.plot(results_lw['heating_rate_clearsky'],results_lw['altitude']/1e3,color='b')
ax.plot(results_sw['heating_rate_clearsky'],results_sw['altitude']/1e3,color='r')
ax.plot(hrtot,results_sw['altitude']/1e3,color='k')
ax.set_ylim([0,20])
ax.set_xlim([-5,5])
fig.savefig(f"hr_{par.case}.png",dpi=300,bbox_inches='tight')

print('done with pyarts-fluxes')
