# The proper usage is: python3 -u run_arts.py [LW/SW/ALL] > test.log 2>&1 &
import sys
import os
import numpy as np
import FluxSimulator as fsm
from pyarts.arts import convert
from pyarts import xml
import shutil
import pickle

def main(mode):
    # =============================================================================
    # Create a bespoke thermodynamic profile using parameters class and ssm1d
    # =============================================================================
    import parameters
    from ssm1d.atm import get_custom_atm
    # Step 0: Define the case and the resolution
    # specify what gases and CIA-pairs to include (see generate_case)
    #gases     = ['H2O','CO2']
    gases = ['H2O']
    ciapairs  = []
    
    # Declare the absorbing species (in ARTS)
    #lut_description = "DEFAULT"
    lut_description = "H2O_CTM"
    if lut_description == "H2O_CTM":
        lwabs_species = ["H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400"]
        swabs_species = ["H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400"]
    elif lut_description == "DEFAULT":
        lwabs_species = [
                "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
                "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
                "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                "CO2, CO2-CKDMT252",
                "CH4",
                "O3",
                "O3-XFIT",
                ]
        swabs_species = [
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
            "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "CO2, CO2-CKDMT252",
            "CH4",
            "O3",
            "O3-XFIT",
        ]
    else:
        print("Please choose a valid lut_description")
        sys.exit(1)

    # dynamically create an argument dictionary
    def generate_args(exp, gases, ciapairs, **thermo):
        return {'exp': exp, **{f'gas{i+1}': gas for i, gas in enumerate(gases)}, 'valid_ciapairs': ciapairs, **thermo}
    #args = generate_args('earth',gases,ciapairs,RHs=0.75,RHmid=0.54,RHtrp=0.75,uniform=1,Ts=298.4,Tmid=250,Ttrp=200) 
    args = generate_args('earth',gases,ciapairs,RHs=0.75,RHmid=0.54,RHtrp=0.75,uniform=1,Ts=298.5,Tmid=250,Ttrp=200)
        
    # create a class instance and generate an RFM case from argument dictionary (300K = 7.1 K/km; 315K = 5.0 K/km)
    par = parameters.parameters(Gamma=6.5e-3,z=np.arange(0,2.0e4,1))
    par.generate_case(**args)

    # vertical resolution
    RFM      = np.arange(par.z[0],par.z[-1],1e2)

    # default dataset
    # z ~ m, p ~ Pa, T ~ K, Gamma ~ K/m, RH~unitless, hr~W/m3, srhr~cm*W/m3
    dataset  = ({'RFM':{},'ARTS':{}}) 

    # Step 1: Generate custom atmospheric profiles (p,T,z,x) at RFM and RFMi resolution
    gca_gases = {
    'H2O': Gas('H2O', M=par.MH2O, cpmol=par.cpmolH2O),  # special: computed from RH
    'N2' : Gas('N2',  M=par.MN2,  cpmol=par.cpmolN2,  xdry=None, fill_remainder=True),
    'O2' : Gas('O2',  M=par.MO2,  cpmol=par.cpmolO2,  xdry=par.xO2), 
    'CO2' : Gas('CO2',  M=par.MCO2,  cpmol=par.cpmolCO2,  xdry=par.xCO2)
    }
    dat1 = get_custom_atm(par, RFM, gca_gases)
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
    #data_path = f"../atmdata/{setup_name}/"
    data_path = f"/u/home/f/fspauldi/pyarts-fluxes/atmdata/{setup_name}/"

    # generate the directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    atm_field = fsm.generate_gridded_field_from_profiles(
        dataset['RFM']['p'], dataset['RFM']['T'], z_field=dataset['RFM']['z'], gases={f"{gases[0]}":dataset['RFM'][f'x{gases[0]}']}, particulates={}
    )

    atm_field.savexml(f'{data_path}/atm_{par.case}.xml','ascii',True)
    shutil.copy(f"/u/home/f/fspauldi/pyarts-fluxes/atmdata/single_atmosphere/aux_earth.xml", f"{data_path}/aux_{par.case}.xml")


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
    #max_wvn_sw = 2010. # [cm^-1]
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
    #max_wvn = 3210  # [cm^-1]
    max_wvn = 1500  # [cm^-1]
    #max_wvn = 15 # [cm^-1]
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
    #surface_temperature = aux[0]  # [K]
    #surface_temperature = 301  # [K]
    surface_temperature = par.Ts # [K]

    # surface reflectivity
    surface_reflectivity_sw = 0.3
    surface_reflectivity_lw = 0.05

    # sun position
    # note: the location of the sun in the sky is specified by the geographical location
    # at which the sun is at zenith. If geographical position of observer is at the equator,
    # then a zenith angle of 30 degrees corresponds to a sun_pos latitude of 30 degrees.  
    sun_pos = [1.495978707e11, 71.0, 0.0]  # [m], [lat; deg], [lon; deg]

    lw_lut_path = f"/u/scratch/f/fspauldi/lut_cache/{lut_description}/LW"
    sw_lut_path = f"/u/scratch/f/fspauldi/lut_cache/{lut_description}/SW"
    if not os.path.exists(lw_lut_path):
        os.makedirs(lw_lut_path, exist_ok=True)
    if not os.path.exists(sw_lut_path):
        os.makedirs(sw_lut_path, exist_ok=True)

    #if not os.path.exists(os.path.join(os.getcwd(), f"datasets/dataset-{par.case}.pkl")):
    #lwcache = os.path.join(os.getcwd(), f"cache/{par.case}_LW")
    #swcache = os.path.join(os.getcwd(), f"cache/{par.case}_SW")
    #if not os.path.exists(lwcache) or not os.path.exists(swcache):
    if 1:
        # =============================================================================
        # longwave simulation
        # =============================================================================

        if (mode == "LW" or mode == "ALL"):
            LW_flux_simulator = fsm.FluxSimulator(setup_name + "_LW")
            LW_flux_simulator.ws.f_grid = f_grid_lw

            # set an optional alternative path to the LUT
            #lut_path = os.path.join(os.getcwd(), "cache", "earth-H2O-CTM_LW") 
            LW_flux_simulator.set_paths(lut_path=lw_lut_path)
            
            # set the absorbing species
            LW_flux_simulator.set_species(lwabs_species)

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
        if (mode == "SW" or mode == "ALL"):
            SW_flux_simulator = fsm.FluxSimulator(setup_name + "_SW")
            SW_flux_simulator.ws.f_grid = f_grid_sw
            SW_flux_simulator.emission = 0
            SW_flux_simulator.gas_scattering = True
            SW_flux_simulator.set_sun(sun_pos)

            # set an optional alternative path to the LUT
            #lut_path = os.path.join(os.getcwd(), "cache", "earth-H2O-CTM_SW") 
            SW_flux_simulator.set_paths(lut_path=sw_lut_path)
            
            # set the absorbing species
            SW_flux_simulator.set_species(swabs_species)

            results_sw = SW_flux_simulator.flux_simulator_single_profile(
                atm,
                surface_temperature,
                surface_altitude,
                surface_reflectivity_sw,
                geographical_position=[lat, lon],
            )
 
        # =============================================================================
        # save the data
        # =============================================================================
        bands = ['LW','SW']
        keys  = ['heating_rate_clearsky', 'altitude']
        for band in bands:
            dataset['ARTS'][band] = {}
            for key in keys:
                if band=='LW':
                    results = results_lw 
                else:
                    results = results_sw
                dataset['ARTS'][band][key] = results[key]
        pfile = os.path.join(os.getcwd(), f"datasets/dataset-{par.case}.pkl")
        with open(pfile,'wb') as f:
            pickle.dump(dataset,f)

        # Save to netcdf (safe)
        ncfile = os.path.join(os.getcwd(), f"datasets/dataset-{par.case}.nc")    
        from save import save_dataset_to_netcdf
        save_dataset_to_netcdf(
	    dataset,
	    outfile=ncfile,
            case=par.case,
	)

    # =============================================================================
    # plot the result
    # =============================================================================

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(12,6),sharey=True)
    data = openpickle(os.path.join(os.getcwd(), f"datasets/dataset-{par.case}.pkl"))
    results_lw = data['ARTS']['LW']
    results_sw = data['ARTS']['SW']
    hrtot = results_lw['heating_rate_clearsky']+results_sw['heating_rate_clearsky']
    ax[0].plot(results_lw['heating_rate_clearsky'],results_lw['altitude']/1e3,color='b')
    ax[0].plot(results_sw['heating_rate_clearsky'],results_sw['altitude']/1e3,color='r')
    ax[0].plot(hrtot,results_sw['altitude']/1e3,color='k')
    ax[0].set_ylim([0,20])
    ax[0].set_xlim([-5,5])

    pfile = os.path.join(os.getcwd(), f"datasets/dataset-{par.case}.pkl")
    data  = get_csc_from_arts(pfile,par)
    ax[1].plot(data['CSC_lw'],results_lw['altitude']/1e3,color='b')
    ax[1].plot(data['CSC_sw'],results_sw['altitude']/1e3,color='r')
    ax[1].plot(data['CSC'],results_sw['altitude']/1e3,color='k')
    ax[1].set_ylim([0,20])
    ax[1].set_xlim(-0.25,0.75)
    fig.savefig(f"hr_{par.case}.png",dpi=300,bbox_inches='tight')

    print('done with pyarts-fluxes')

# =============================================================================
def openpickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_csc_from_arts(fname,par):
    # Unpack the inputs
    data = openpickle(fname)
    heights = data['RFM']['z']
    rho     = data['RFM']['rho']
    sigma   = par.ggr/par.cpa - data['RFM']['Gamma']
    tmp_lw  = -data['ARTS']['LW']['heating_rate_clearsky']
    tmp_sw  = -data['ARTS']['SW']['heating_rate_clearsky']
    H_lw    = par.cpa*rho*tmp_lw/86400   # K/day->K/s->W/m3 
    H_sw    = par.cpa*rho*tmp_sw/86400   # K/day->K/s->W/m3
    H       = H_lw + H_sw
    dzH_lw  = np.gradient(H_lw,heights) # W/m4
    dzH_sw  = np.gradient(H_sw,heights) # W/m4
    dzH     = np.gradient(H,heights) # W/m4
    #############################################################
    CSC_lw  = -dzH_lw/(par.cpa*rho*sigma)*86400 # 1/s->1/day  
    CSC_sw  = -dzH_sw/(par.cpa*rho*sigma)*86400 # 1/s->1/day
    CSC     = -dzH/(par.cpa*rho*sigma)*86400 # 1/s->1/day
    #############################################################
    return {'CSC':CSC,'CSC_lw':CSC_lw,'CSC_sw':CSC_sw}

# =============================================================================

def init():
    if len(sys.argv) != 2:
        print("Usage: python run_arts.py [LW|SW|ALL]")
        sys.exit(1)

    mode = sys.argv[1].upper()

    if mode not in ["LW", "SW", "ALL"]:
        print("Error: input must be 'LW' or 'SW' or 'ALL'")
        sys.exit(1)
    return mode

# =============================================================================

if __name__ == "__main__":
    mode = init()
    main(mode)

