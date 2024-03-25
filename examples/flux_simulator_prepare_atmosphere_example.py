#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:21:58 2024

@author: Manfred Brath

Example script to generate a simple atmosphere for the flux simulator.
The atmosphere is generated from simple profiles for pressure, temperature,
H2O, CO2 and LWC. The profiles are used to generate a ARTS GriddedField4 object
which is needed for the flux simulator. The generated atmosphere can be saved to a
file in ARTS XML format. 

"""
# %%

import numpy as np
from pyarts import xml
from FluxSimulator import generate_gridded_field_from_profiles

# %% generate example atmosphere needed for standard LUT
# This atmosphere is not intended to be fully realistic, but to be simply
# an example for the calculation of LUT.

# set pressure grid
nlev = 80
pressure_profile = np.linspace(1000e2, 1e2, nlev)

# create water vapor profile
# Water vapor is simply define by a 1st order
# polynomial in log-log space
# log h2o = a + b * log pressure
b = 4
a = -6 - b * 4
logH2O = a + b * np.log10(pressure_profile)
H2O_profile = 10**logH2O

# create temperature profile
# Temperature is simply define by a 1st order
# polynomial of log pressure
# T = a + b * log pressure
# For pressure < 100 hPa, the temperature is set to 200 K
b = 100
a = 200 - b * 4
temperature_profile = a + b * np.log10(pressure_profile)
temperature_profile[pressure_profile < 100e2] = (
    200  # set temperature to 200 K below 100 hPa
)

# CO2 vmr value
CO2 = 400e-6  # [vmr]

# LWC
LWC_profile = np.zeros_like(pressure_profile)
LWC_profile[10:14] = 1e-4


# %% generate gridded field from profiles
# This function generates a ARTS GriddedField4 from the given profiles 
# which is needed for the flux simulator. The function takes the profiles
# for pressure, temperature, H2O, CO2 and LWC as input and returns a ARTS
# GriddedField4 object. The z_field can be set to None, in this case the
# z_field is generated from the pressure profile internally. But it is not 
# recommended if you need accurate height information. 


atm_field = generate_gridded_field_from_profiles(
    pressure_profile,
    temperature_profile,
    gases={"H2O": H2O_profile, "CO2": CO2},
    particulates={"LWC-mass_density": LWC_profile},
    z_field=None
)

# %% save the generated atmosphere to a file
# The generated atmosphere can be saved to a file with the following function.
# The function takes the GriddedField4 object and the filename as input and
# saves the atmosphere to a file.

atm_filename = "atm_single_atmosphere.xml"
xml.save(atm_field, atm_filename)
