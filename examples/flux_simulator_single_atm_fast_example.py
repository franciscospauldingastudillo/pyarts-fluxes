#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:21:58 2024

@author: Manfred Brath

Example script to demonstrate the usage of the flux_simulator_module 
using the fast scheme for a single atmosphere simulations.

"""
import os
import numpy as np
import FluxSimulator as fsm
from pyarts.arts import convert
from pyarts import xml


# =============================================================================
# Paths and constants
# =============================================================================


setup_name = "single_atmosphere_fast_exam"
data_path = f"../atmdata/single_atmosphere/"
atm_filename = "atm_idx9108_lat10.975_lon36.975.xml"
aux_filename = "aux_idx9108_lat10.975_lon36.975.xml"
# atm_filename = "atm_idx9919_lat20.975_lon-141.025.xml"
# aux_filename = "aux_idx9919_lat20.975_lon-141.025.xml"
# atm_filename = "atm_mean.xml"
# aux_filename = "aux_mean.xml"

# =============================================================================
# Set frequency grids
# =============================================================================

min_wavelength_sw = 3e-7  # [m]
max_wavelength_sw = 5e-6  # [m]
n_freq_sw = 200

wvl = np.linspace(min_wavelength_sw, max_wavelength_sw, n_freq_sw)  # [m]
f_grid_sw = convert.wavelen2freq(wvl[::-1]) # [Hz]


min_wvn = 10 # [cm^-1]
max_wvn = 3210 # [cm^-1]
n_freq_lw = 200
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
surface_altitude = aux[1] # [m]

# surface temperature
surface_temperature = aux[0] # [K]

# surface reflectivity
surface_reflectivity_sw = 0.3
surface_reflectivity_lw = 0.05

# sun position
sun_pos = [1.495978707e11, 0.0, 36.0] # [m], [deg], [deg]



# =============================================================================
# longwave simulation
# =============================================================================

LW_flux_simulator = fsm.FluxSimulator(setup_name + "_LW")
LW_flux_simulator.ws.f_grid.readxml('planets/Earth/Optimized-Flux-Frequencies/LW-flux-optimized-f_grid.xml')
LW_flux_simulator.quadrature_weights.readxml('planets/Earth/Optimized-Flux-Frequencies/LW-flux-optimized-quadrature_weights.xml')

results_lw = LW_flux_simulator.flux_simulator_single_profile(
    atm,
    surface_temperature,
    surface_altitude,
    surface_reflectivity_sw,
    geographical_position=[lat, lon],
    sun_pos=[],
)


# =============================================================================
# shortwave simulation
# =============================================================================

SW_flux_simulator = fsm.FluxSimulator(setup_name + "_SW")
SW_flux_simulator.ws.f_grid.readxml('planets/Earth/Optimized-Flux-Frequencies/SW-flux-optimized-f_grid.xml')
SW_flux_simulator.quadrature_weights.readxml('planets/Earth/Optimized-Flux-Frequencies/SW-flux-optimized-quadrature_weights.xml')
SW_flux_simulator.emission = 0
SW_flux_simulator.gas_scattering = True

results_sw = SW_flux_simulator.flux_simulator_single_profile(
    atm,
    surface_temperature,
    surface_altitude,
    surface_reflectivity_sw,
    geographical_position=[lat, lon],
    sun_pos=sun_pos,
)
