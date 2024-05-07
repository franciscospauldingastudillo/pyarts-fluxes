#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module contains the class FluxSimulator. This class is used to calculate the fluxes and heating rates
for a given atmosphere. The atmosphere can be defined by a GriddedField4 object or by a list of profiles.
The fluxes are calculated using the ARTS radiative transfer model.
All quantities are defined in SI units if not stated otherwise. For example, the unit of fluxes is W/m^2 or the
unit of frequency is Hz.



@author: Manfred Brath
"""
# %%
import os
import numpy as np
from pyarts import cat, xml, arts, version
from pyarts.workspace import Workspace
from . import _flux_simulator_agendas as fsa

# %%


class FluxSimulationConfig:
    """
    This class defines the basic setup for the flux simulator.
    """

    def __init__(self, setup_name, catalog_version = None):
        """
        Parameters
        ----------
        setup_name : str
            Name of the setup. This name is used to create the directory for the LUT.

        Returns
        -------
        None.
        """

        #check version
        version_min=[2,6,2]
        v_list=version.split('.')
        major=int(v_list[0])==version_min[0]
        minor=int(v_list[1])==version_min[1]
        patch=int(v_list[2])>=version_min[2]
        
        if not major or not minor or not patch:
            raise ValueError(f"Please use pyarts version >= {'.'.join(str(i) for i in version_min)}.")

        self.setup_name = setup_name

        # set default species
        self.species = [
            "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
            "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
            "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
            "CO2, CO2-CKDMT252",
            "CH4",
            "O3",
            "O3-XFIT",
        ]

        # set some default values for some well mixed species
        self.well_mixed_species_defaults = {}
        self.well_mixed_species_defaults["CO2"] = 415e-6
        self.well_mixed_species_defaults["CH4"] = 1.8e-6
        self.well_mixed_species_defaults["O2"] = 0.21
        self.well_mixed_species_defaults["N2"] = 0.78

        # set default paths
        self.catalog_version = catalog_version
        self.basename_scatterer = os.path.join(
            os.path.dirname(__file__), "..", "..", "scattering_data"
        )

        # Be default the solar spectrum is set to the May 2004 spectrum
        # Other options are "Blackbody" or a path to a spectrum file
        self.sunspectrumtype = "SpectrumMay2004"

        # set default parameters
        self.Cp = 1.0057e03  # specific heat capacity of dry air [Jkg^{-1}K^{-1}] taken from AMS glossary
        self.nstreams = 10
        self.emission = 1
        self.quadrature_weights = np.array([])

        # set if allsky or clearsky
        self.allsky = False

        # set if gas scattering is used
        self.gas_scattering = False

        # set default LUT path
        self.lut_path = os.path.join(os.getcwd(), "cache", setup_name)
        self.lutname_fullpath = os.path.join(self.lut_path, "LUT.xml")

        
        cat.download.retrieve(
            version=self.catalog_version, verbose=True
        )

    def generateLutDirectory(self, alt_path=None):
        """
        This function creates the directory for the LUT.

        Parameters
        ----------
        alt_path : str, optional
            Alternative path for the LUT. The default is None.

        Returns
        -------
        None.
        """
        if alt_path is not None:
            self.lut_path = alt_path
            self.lutname_fullpath = os.path.join(self.lut_path, "LUT.xml")
        os.makedirs(self.lut_path, exist_ok=True)

    def set_paths(
        self,
        basename_scatterer=None,
        lut_path=None,
    ):
        """
        This function sets some paths. If a path is not given, the default path is used.
        This function is needed only if you want to use different paths than the default paths.

        Parameters
        ----------

        basename_scatterer : str, optional
            Path to the scatterer. The default is None.
        lut_path : str, optional
            Path to the LUT. The default is None.

        Returns
        -------
        None.

        """

        if basename_scatterer is not None:
            self.basename_scatterer = basename_scatterer

        if lut_path is not None:
            self.generateLutDirectory(lut_path)

    def get_paths(self):
        """
        This function returns the paths as a dictionary.

        Returns
        -------
        Paths : dict
            Dictionary containing the paths.
        """

        Paths = {}
        Paths["basename_scatterer"] = self.basename_scatterer
        Paths["sunspectrumpath"] = self.sunspectrumtype
        Paths["lut_path"] = self.lut_path
        Paths["lutname_fullpath"] = self.lutname_fullpath

        return Paths

    def print_paths(self):
        """
        This function prints the paths.

        Returns
        -------
        None.
        """

        print("basename_scatterer: ", self.basename_scatterer)
        print("lut_path: ", self.lut_path)
        print("lutname_fullpath: ", self.lutname_fullpath)

    def print_config(self):
        """
        This function prints the setup.

        Returns
        -------
        None.
        """

        print("setup_name: ", self.setup_name)
        print("species: ", self.species)
        print("Cp: ", self.Cp)
        print("nstreams: ", self.nstreams)
        print("emission: ", self.emission)
        print("quadrature_weights: ", self.quadrature_weights)
        print("allsky: ", self.allsky)
        print("gas_scattering: ", self.gas_scattering)
        print("sunspectrumtype: ", self.sunspectrumtype)
        self.print_paths()


class FluxSimulator(FluxSimulationConfig):

    def __init__(self, setup_name, catalog_version = None):
        """
        This class defines the ARTS setup.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        super().__init__(setup_name, catalog_version = catalog_version)

        # start ARTS workspace
        self.ws = Workspace()
        self.ws.verbositySetScreen(level=2)
        self.ws.verbositySetAgenda(level=0)

        # Set stoke dimension
        self.ws.IndexSet(self.ws.stokes_dim, 1)

        # Create my defined agendas in ws
        (
            self.ws,
            self.gas_scattering_agenda_list,
            self.surface_agenda_list,
            self.pnd_agenda_list,
        ) = fsa.create_agendas_in_ws(self.ws, pnd_agendas=True)

        self.ws = fsa.set_pnd_agendas_SB06(self.ws)
        self.ws = fsa.set_pnd_agendas_MY05(self.ws)
        self.ws = fsa.set_pnd_agendas_CG(self.ws)

        # Initialize scattering variables
        self.ws.ScatSpeciesInit()
        self.ws.ArrayOfArrayOfScatteringMetaDataCreate("scat_meta_temp")
        self.ws.ArrayOfArrayOfSingleScatteringDataCreate("scat_data_temp")

        # select/define agendas
        # =============================================================================

        self.ws.PlanetSet(option="Earth")

        self.ws.gas_scattering_agenda = fsa.gas_scattering_agenda__Rayleigh
        self.ws.iy_main_agendaSet(option="Clearsky")
        self.ws.iy_space_agendaSet(option="CosmicBackground")
        self.ws.iy_cloudbox_agendaSet(option="LinInterpField")
        self.ws.water_p_eq_agendaSet()
        self.ws.ppath_step_agendaSet(option="GeometricPath")
        self.ws.ppath_agendaSet(option="FollowSensorLosPath")

        # define environment
        # =============================================================================

        self.ws.AtmosphereSet1D()

        # Number of Stokes components to be computed
        #
        self.ws.IndexSet(self.ws.stokes_dim, 1)

        # No jacobian calculations
        self.ws.jacobianOff()

        # set absorption species
        self.ws.abs_speciesSet(species=self.species)

    def set_species(self, species):
        """
        This function sets the gas absorption species.

        Parameters
        ----------
        species : list
            List of species.

        Returns
        -------
        None.
        """

        self.species = species
        self.ws.abs_species = self.species

    def get_species(self):
        """
        This function returns the gas absorption species.

        Returns
        -------
        list
            List of species.
        """

        return self.ws.abs_species

    def check_species(self):
        """
        This function checks if all species are included in the atm_fields_compact
        that are defined in abs_species. If not, the species are added with the default
        values from well_mixed_species_defaults.
        A ValueError is raised if a species is not included in the atm_fields_compact and
        not in well_mixed_species_defaults.

        Returns
        -------
        None.
        """

        atm_grids = self.ws.atm_fields_compact.value.grids[0]

        # Get species of atm-field
        atm_species = [
            str(tag).split("-")[1] for tag in atm_grids if "abs_species" in str(tag)
        ]

        # Get species from defined abs_species
        abs_species = self.get_species().value
        abs_species = [str(tag).split("-")[0] for tag in abs_species]

        for abs_species_i in abs_species:

            if abs_species_i not in atm_species:

                # check for default
                if abs_species_i in self.well_mixed_species_defaults.keys():

                    self.ws.atm_fields_compactAddConstant(
                        self.ws.atm_fields_compact,
                        f"abs_species-{abs_species_i}",
                        self.well_mixed_species_defaults[abs_species_i],
                    )

                    print(
                        f"{abs_species_i} data not included in atmosphere data\n"
                        f"I will use default value {self.well_mixed_species_defaults[abs_species_i]}"
                    )

                else:

                    self.ws.atm_fields_compactAddConstant(
                        self.ws.atm_fields_compact,
                        f"abs_species-{abs_species_i}",
                        0.,
                    )

                    print(
                        f"{abs_species_i} data not included in atmosphere data\n"
                        f"and it is not in well_mixed_species_defaults\n"
                        f"I will add this species with value 0."
                    )

    def define_particulate_scatterer(
        self,
        hydrometeor_type,
        pnd_agenda,
        scatterer_name,
        moments,
        scattering_data_folder=None,
    ):
        """
        This function defines a particulate scatterer.

        Parameters
        ----------
        hydrometeor_type : str
            Hydrometeor type.
        pnd_agenda : str
            PND agenda.
        scatterer_name : str
            Scatterer name.
        moments : list
            Moments of psd.
        scattering_data_folder : str
            Scattering data folder.

        Returns
        -------
        None.

        """

        if scattering_data_folder is None:
            scattering_data_folder = self.basename_scatterer

        self.ws.StringCreate("species_id_string")
        self.ws.StringSet(self.ws.species_id_string, hydrometeor_type)
        self.ws.ArrayOfStringSet(
            self.ws.pnd_agenda_input_names,
            [f"{hydrometeor_type}-{moment}" for moment in moments],
        )
        self.ws.Append(self.ws.pnd_agenda_array, eval(f"self.ws.{pnd_agenda}"))
        self.ws.Append(self.ws.scat_species, self.ws.species_id_string)
        self.ws.Append(
            self.ws.pnd_agenda_array_input_names, self.ws.pnd_agenda_input_names
        )

        ssd_name = os.path.join(scattering_data_folder, f"{scatterer_name}.xml")
        self.ws.ReadXML(self.ws.scat_data_temp, ssd_name)
        smd_name = os.path.join(scattering_data_folder, f"{scatterer_name}.meta.xml")
        self.ws.ReadXML(self.ws.scat_meta_temp, smd_name)
        self.ws.Append(self.ws.scat_data_raw, self.ws.scat_data_temp)
        self.ws.Append(self.ws.scat_meta, self.ws.scat_meta_temp)

        self.allsky = True

    def readLUT(self, F_grid_from_LUT=False, fmin=0, fmax=np.inf):
        """
        Reads the Look-Up Table (LUT).

        Parameters:
            F_grid_from_LUT (bool, optional): Flag indicating whether to use the f_grid from the LUT.
                                              Defaults to False.
            fmin (float, optional): Minimum frequency value to read. Defaults to 0.
            fmax (float, optional): Maximum frequency value to read. Defaults to np.inf.

        Returns:
            None
        """

        self.ws.Touch(self.ws.abs_lines_per_species)
        self.ws.ReadXML(self.ws.abs_lookup, self.lutname_fullpath)

        if F_grid_from_LUT == True:
            print("Using f_grid from LUT")
            f_grid = np.array(self.ws.abs_lookup.value.f_grid.value)

            f_grid = f_grid[fmin < f_grid]
            f_grid = f_grid[f_grid < fmax]

            self.ws.f_grid = f_grid
        else:
            f_grid = np.array(self.ws.f_grid.value)

        self.ws.abs_lookupAdapt()
        self.ws.lbl_checked = 1

    def get_lookuptableWide(
        self,
        t_min=150.0,
        t_max=350.0,
        p_step=0.5,
        lines_speedup_option="None",
        F_grid_from_LUT=False,
        cutoff=True,
        fmin=0,
        fmax=np.inf,
        recalc=False,
    ):
        """
        This function calculates the LUT using the wide setup.

        Parameters
        ----------
        t_min : float
            Minimum temperature.
        t_max : float
            Maximum temperature.
        p_step : float
            Pressure step.
        lines_speedup_option : str
            Lines speedup option.
        F_grid_from_LUT : bool
            If True, the frequency grid is taken from the LUT.
        cutoff : bool
            If True, cutoff is used.
        fmin : float
            Minimum frequency.
        fmax : float
            Maximum frequency.
        recalc : bool
            If True, the LUT is recalculated.

        Returns
        -------
        None.

        """

        # use saved LUT. recalc only when necessary
        if recalc == False:
            try:
                self.readLUT(F_grid_from_LUT=F_grid_from_LUT, fmin=fmin, fmax=fmax)
                print("...using stored LUT\n")

            # recalc LUT
            except RuntimeError:
                recalc = True

        if recalc == True:
            print("LUT not found or does not fit.\n So, recalc...\n")

            # generate LUT path
            self.generateLutDirectory()

            # read spectroscopic data
            print("...reading data\n")
            self.ws.ReadXsecData(basename="xsec/")
            self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

            if cutoff == True:
                self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

            # setup LUT
            print("...setting up lut\n")
            self.ws.abs_lookupSetupWide(t_min=t_min, t_max=t_max, p_step=p_step)

            # Setup propagation matrix agenda (absorption)
            self.ws.propmat_clearsky_agendaAuto(
                lines_speedup_option=lines_speedup_option
            )

            if cutoff == True:
                self.ws.lbl_checked = 1
            else:
                self.ws.lbl_checkedCalc()

            # calculate LUT
            print("...calculating lut\n")
            self.ws.abs_lookupCalc()

            # save Lut
            self.ws.WriteXML("binary", self.ws.abs_lookup, self.lutname_fullpath)

            print("LUT calculation finished!")

    def get_lookuptable(
        self,
        pressure_profile,
        temperature_profile,
        vmr_profiles,
        p_step=0.05,
        lines_speedup_option="None",
        F_grid_from_LUT=False,
        cutoff=True,
        fmin=0,
        fmax=np.inf,
        recalc=False,
    ):
        """
        This function calculates the LUT using the wide setup.
        It is important that the size of the first dimension of
        vmr_profiles matches the defined species list.
        I cannot check this here, because vmr_profiles is only simple matrix.


        Parameters
        ----------
        pressure_profile : 1Darray
            Pressure profile.
        temperature_profile : 1Darray
            Temperature profile.
        vmr_profiles : 2Darray
            VMR profiles.
        p_step : float
            Pressure step.
        lines_speedup_option : str
            Lines speedup option.
        F_grid_from_LUT : bool
            If True, the frequency grid is taken from the LUT.
        cutoff : bool
            If True, cutoff is used.
        fmin : float
            Minimum frequency.
        fmax : float
            Maximum frequency.
        recalc : bool
            If True, the LUT is recalculated.

        Returns
        -------
        None.

        """

        # use saved LUT. recalc only when necessary
        if recalc == False:
            try:
                self.readLUT(F_grid_from_LUT=F_grid_from_LUT, fmin=fmin, fmax=fmax)
                print("...using stored LUT\n")

            # recalc LUT
            except RuntimeError:
                recalc = True

        if recalc == True:
            print("LUT not found or does not fit.\n So, recalc...\n")

            # check if vmr has the right amout of species
            if np.size(vmr_profiles, 1) != np.size(pressure_profile):
                raise ValueError(
                    "The amount of pressure levels in the vmr_profiles does not match the amount of the pressure levels in the pressure_profile!"
                )

            if np.size(vmr_profiles, 1) != np.size(temperature_profile):
                raise ValueError(
                    "The amount of temperature levels in the vmr_profiles does not match the amount of the temperature levels in the temperature_profile!"
                )

            # put quantities into ARTS
            self.ws.p_grid = pressure_profile
            self.ws.t_field = np.reshape(
                temperature_profile, (len(pressure_profile), 1, 1)
            )
            self.ws.vmr_field = np.reshape(
                vmr_profiles, (len(self.species), len(pressure_profile), 1, 1)
            )

            # generate LUT path
            self.generateLutDirectory()

            # read spectroscopic data
            print("...reading data\n")
            self.ws.ReadXsecData(basename="xsec/")
            self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

            if cutoff == True:
                self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

            # setup LUT
            print("...setting up lut\n")
            self.ws.atmfields_checked = 1
            self.ws.abs_lookupSetup(p_step=p_step)

            # Setup propagation matrix agenda (absorption)
            self.ws.propmat_clearsky_agendaAuto(
                lines_speedup_option=lines_speedup_option
            )

            if cutoff == True:
                self.ws.lbl_checked = 1
            else:
                self.ws.lbl_checkedCalc()

            # calculate LUT
            print("...calculating lut\n")
            self.ws.abs_lookupCalc()

            # save Lut
            self.ws.WriteXML("binary", self.ws.abs_lookup, self.lutname_fullpath)

            print("LUT calculation finished!")

    def flux_simulator_single_profile(
        self,
        atm,
        T_surface,
        z_surface,
        surface_reflectivity,
        geographical_position=np.array([]),
        sun_pos=np.array([]),
    ):
        """
        This function calculates the fluxes and heating rates for a single atmosphere.
        The atmosphere is defined by the ARTS GriddedField4 atm.

        Parameters
        ----------
        f_grid : 1Darray
            Frequency grid.
        atm : GriddedField4
            Atmosphere.
        T_surface : float
            Surface temperature.
        z_surface : float
            Surface altitude.
        surface_reflectivity : float or 1Darray
            Surface reflectivity.
        geographical_position : 1Darray, default is np.array([])
            Geographical position.
        sun_pos : 1Darray, default is np.array([])
            Sun position.

        Returns
        -------
        results : dict
            Dictionary containing the results.
            results["flux_clearsky_up"] : 1Darray
                Clearsky flux up.
            results["flux_clearsky_down"] : 1Darray
                Clearsky flux down.
            results["spectral_flux_clearsky_up"] : 2Darray
                Clearsky spectral flux up.
            results["spectral_flux_clearsky_down"]  : 2Darray
                Clearsky spectral flux down.
            results["heating_rate_clearsky"] : 1Darray
                Clearsky heating rate in K/d.
            results["pressure"] : 1Darray
                Pressure.
            results["altitude"] : 1Darray
                Altitude.
            results["f_grid"] : 1Darray
                Frequency grid.
            results["flux_allsky_up"] : 1Darray, optional
                Allsky flux up.
            results["flux_allsky_down"] : 1Darray, optional
                Allsky flux down.
            results["spectral_flux_allsky_up"] : 2Darray, optional
                Allsky spectral flux up.
            results["spectral_flux_allsky_down"]  : 2Darray, optional
                Allsky spectral flux down.
            results["heating_rate_allsky"] : 1Darray, optional
                Allsky heating rate in K/d.


        """

        # define environment
        # =============================================================================

        if len(sun_pos) > 0:
            # set sun source
            if self.sunspectrumtype == "Blackbody":
                self.ws.sunsAddSingleBlackbody(
                    distance=sun_pos[0], latitude=sun_pos[1], longitude=sun_pos[2]
                )
            elif len(self.sunspectrumtype) > 0:
                sunspectrum=arts.GriddedField2()
                if self.sunspectrumtype == "SpectrumMay2004":
                    sunspectrum.readxml('star/Sun/solar_spectrum_May_2004.xml')
                else:
                    sunspectrum.readxml(self.sunspectrumtype)


                self.ws.sunsAddSingleFromGrid(
                    sun_spectrum_raw=sunspectrum,
                    temperature=0,
                    distance=sun_pos[0],
                    latitude=sun_pos[1],
                    longitude=sun_pos[2],
                )
            else:
                print("No sun source defined!")
                print("Setting suns off!")
                self.ws.sunsOff()

        else:
            self.ws.sunsOff()

        # prepare atmosphere
        self.ws.atm_fields_compact = atm
        self.check_species()
        self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

        # set absorption
        # =============================================================================

        print("setting up absorption...\n")

        # Calculate or load LUT
        self.get_lookuptableWide()

        # Use LUT for absorption
        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        # setup
        # =============================================================================

        # surface altitudes
        self.ws.z_surface = [[z_surface]]

        # surface temperatures
        self.ws.surface_skin_t = T_surface

        # set geographical position
        self.ws.lat_true = [geographical_position[0]]
        self.ws.lon_true = [geographical_position[1]]

        # surface reflectivities
        self.ws.surface_scalar_reflectivity = [surface_reflectivity]

        print("starting calculation...\n")

        # no sensor
        self.ws.sensorOff()

        # set cloudbox to full atmosphere
        self.ws.cloudboxSetFullAtm()

        # set gas scattering on or off
        if self.gas_scattering == False:
            self.ws.gas_scatteringOff()
        else:
            self.ws.gas_scattering_do=1

        if self.allsky:
            self.ws.scat_dataCalc(interp_order=1)
            self.ws.Delete(self.ws.scat_data_raw)
            self.ws.scat_dataCheck(check_type="all")

            self.ws.pnd_fieldCalcFromParticleBulkProps()
            self.ws.scat_data_checkedCalc()
        else:

            if len(self.ws.scat_species.value) > 0:
                print(
                    ("You have define scattering species for a clearsky simulation.\n")
                    + (
                        "Since they are not used we have to erase the scattering species!\n"
                    )
                )
                self.ws.scat_species = []
            self.ws.scat_data_checked = 1
            self.ws.Touch(self.ws.scat_data)
            self.ws.pnd_fieldZero()

        self.ws.atmfields_checkedCalc()
        self.ws.atmgeom_checkedCalc()
        self.ws.cloudbox_checkedCalc()

        # Set specific heat capacity
        self.ws.Tensor3SetConstant(
            self.ws.specific_heat_capacity,
            len(self.ws.p_grid.value),
            1,
            1,
            self.Cp,
        )

        self.ws.StringCreate("Text")
        self.ws.StringSet(self.ws.Text, "Start disort")
        self.ws.Print(self.ws.Text, 0)

        aux_var_allsky=[]
        if self.allsky:
            # allsky flux
            # ====================================================================================

            self.ws.spectral_irradiance_fieldDisort(
                nstreams=self.nstreams,
                Npfct=-1,
                emission=self.emission,
            )

            self.ws.StringSet(self.ws.Text, "disort finished")
            self.ws.Print(self.ws.Text, 0)

            # get auxilary varibles
            if len(self.ws.disort_aux_vars.value):
                for i in range(len(self.ws.disort_aux_vars.value)):
                    aux_var_allsky.append(self.ws.disort_aux.value[i][:]*1.)

            spec_flux = self.ws.spectral_irradiance_field.value[:, :, 0, 0, :] * 1.0

            self.ws.RadiationFieldSpectralIntegrate(
                self.ws.irradiance_field,
                self.ws.f_grid,
                self.ws.spectral_irradiance_field,
                self.quadrature_weights,
            )
            flux = np.squeeze(self.ws.irradiance_field.value.value) * 1.0

            self.ws.heating_ratesFromIrradiance()

            heating_rate = np.squeeze(self.ws.heating_rates.value) * 86400  # K/d

        # clearsky flux
        # ====================================================================================

        self.ws.pnd_fieldZero()
        self.ws.spectral_irradiance_fieldDisort(
            nstreams=self.nstreams,
            Npfct=-1,
            emission=self.emission,
        )

        # get auxilary varibles
        aux_var_clearsky=[]
        if len(self.ws.disort_aux_vars.value):
            for i in range(len(self.ws.disort_aux_vars.value)):
                aux_var_clearsky.append(self.ws.disort_aux.value[i][:]*1.)

        spec_flux_cs = self.ws.spectral_irradiance_field.value[:, :, 0, 0, :] * 1.0

        self.ws.RadiationFieldSpectralIntegrate(
            self.ws.irradiance_field,
            self.ws.f_grid,
            self.ws.spectral_irradiance_field,
            self.quadrature_weights,
        )
        flux_cs = np.squeeze(self.ws.irradiance_field.value.value)

        self.ws.heating_ratesFromIrradiance()
        heating_rate_cs = np.squeeze(self.ws.heating_rates.value) * 86400  # K/d

        # results
        # ====================================================================================

        results = {}

        results["flux_clearsky_up"] = flux_cs[:, 1]
        results["flux_clearsky_down"] = flux_cs[:, 0]
        results["spectral_flux_clearsky_up"] = spec_flux_cs[:, :, 1]
        results["spectral_flux_clearsky_down"] = spec_flux_cs[:, :, 0]
        results["heating_rate_clearsky"] = heating_rate_cs
        results["pressure"] = self.ws.p_grid.value[:]
        results["altitude"] = self.ws.z_field.value[:, 0, 0]
        results["f_grid"] = self.ws.f_grid.value[:]
        results["aux_var_clearsky"] = aux_var_clearsky

        if self.allsky:
            results["flux_allsky_up"] = flux[:, 1]
            results["flux_allsky_down"] = flux[:, 0]
            results["spectral_flux_allsky_up"] = spec_flux[:, :, 1]
            results["spectral_flux_allsky_down"] = spec_flux[:, :, 0]
            results["heating_rate_allsky"] = heating_rate
            results["aux_var_allsky"] = aux_var_allsky

        return results

    def flux_simulator_batch(
        self,
        atmospheres,
        surface_tempratures,
        surface_altitudes,
        surface_reflectivities,
        geographical_positions,
        sun_positions,
        start_index=0,
        end_index=-1,
    ):
        """
        This function calculates the fluxes for a batch of atmospheres.
        The atmospheres are defined by an array of atmospheres.

        Parameters
        ----------
        f_grid : 1Darray
            Frequency grid.
        atmospheres : ArrayOfGriddedField4
            Batch of atmospheres.
        surface_tempratures : 1Darray
            Surface temperatures.
        surface_altitudes : 1Darray
            Surface altitudes.
        surface_reflectivities : 1Darray
            Surface reflectivities with each row either one element list of a list with the length of f_grid.
        geographical_positions : 2Darray
            Geographical positions with each row containing lat and lon.
        sun_positions : 2Darray
            Sun positions with each row conating distance sun earth, zenith latitude and zenith longitude.
        start_index : int, default is 0
            Start index of batch calculation.
        end_index : int, default is -1
            End index of batch calculation.

        Returns
        -------
        results : dict
            Dictionary containing the results.
            results["array_of_irradiance_field_clearsky"] : 3Darray
                Clearsky irradiance field.
            results["array_of_pressure"] : 2Darray
                Pressure.
            results["array_of_altitude"] : 2Darray
                Altitude.
            results["array_of_latitude"] : 1Darray
                Latitude.
            results["array_of_longitude"] : 1Darray
                Longitude.
            results["array_of_index"] : 1Darray
                Index.
            results["array_of_irradiance_field_allsky"] : 3Darray, optional
                Allsky irradiance field.


        """

        # define environment
        # =============================================================================

        # if len(self.sunspectrumtype) == 0:
        #     raise ValueError("sunspectrumpath not set!")
        # else:
        #     self.ws.GriddedField2Create("sunspectrum")
        #     self.ws.sunspectrum = xml.load(self.sunspectrumtype)
        # set sun source
        # set sun source
        if self.sunspectrumtype == "Blackbody":
            raise ValueError("Blackbody sun not supported for batch !")
        elif len(self.sunspectrumtype) > 0:
            sunspectrum=arts.GriddedField2()
            if self.sunspectrumtype == "SpectrumMay2004":
                sunspectrum.readxml('star/Sun/solar_spectrum_May_2004.xml')
            else:
                sunspectrum.readxml(self.sunspectrumtype) 
            self.ws.GriddedField2Create("sunspectrum")
            self.ws.sunspectrum = sunspectrum
        else:
            raise ValueError("sunspectrumpath not set!")

        # prepare atmosphere
        self.ws.batch_atm_fields_compact = atmospheres

        # list of surface altitudes
        self.ws.ArrayOfMatrixCreate("array_of_z_surface")
        self.ws.array_of_z_surface = [
            np.array([[surface_altitude]]) for surface_altitude in surface_altitudes
        ]

        # list of surface temperatures
        self.ws.VectorCreate("vector_of_T_surface")
        self.ws.vector_of_T_surface = surface_tempratures

        self.ws.MatrixCreate("matrix_of_Lat")
        matrix_of_Lat = np.array([[geo_pos[0]] for geo_pos in geographical_positions])
        self.ws.matrix_of_Lat = matrix_of_Lat

        self.ws.MatrixCreate("matrix_of_Lon")
        matrix_of_Lon = np.array([[geo_pos[1]] for geo_pos in geographical_positions])
        self.ws.matrix_of_Lon = matrix_of_Lon

        # List of surface reflectivities
        self.ws.ArrayOfVectorCreate("array_of_surface_scalar_reflectivity")
        self.ws.array_of_surface_scalar_reflectivity = surface_reflectivities

        # set name of surface probs
        self.ws.ArrayOfStringSet(self.ws.surface_props_names, ["Skin temperature"])

        # list of sun positions
        self.ws.ArrayOfVectorCreate("array_of_sun_positions")
        self.ws.array_of_sun_positions = [sun_pos for sun_pos in sun_positions]

        self.ws.ArrayOfIndexCreate("ArrayOfSuns_Do")
        self.ws.ArrayOfSuns_Do = [
            1 if len(sun_pos) > 0 else 0 for sun_pos in sun_positions
        ]

        # set absorption
        # =============================================================================

        print("setting up absorption...\n")

        # Calculate or load LUT
        self.get_lookuptableWide()

        # Use LUT for absorption
        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        # set gas scattering on or off
        if self.gas_scattering == False:
            self.ws.gas_scatteringOff()
        else:
            self.ws.gas_scattering_do=1

        self.ws.NumericCreate("DummyVariable")
        self.ws.IndexCreate("DummyIndex")
        self.ws.IndexCreate("EmissionIndex")
        self.ws.IndexCreate("NstreamIndex")
        self.ws.StringCreate("Text")
        self.ws.EmissionIndex = int(self.emission)
        self.ws.NstreamIndex = int(self.nstreams)
        self.ws.VectorCreate("quadrature_weights")
        self.ws.quadrature_weights = self.quadrature_weights
        self.ws.NumericCreate("sun_dist")
        self.ws.NumericCreate("sun_lat")
        self.ws.NumericCreate("sun_lon")
        self.ws.VectorCreate("sun_pos")

        print("starting calculation...\n")

        self.ws.IndexSet(self.ws.ybatch_start, start_index)
        if end_index == -1:
            self.ws.IndexSet(self.ws.ybatch_n, len(atmospheres) - start_index)
            len_of_output = len(atmospheres) - start_index
        else:
            self.ws.IndexSet(self.ws.ybatch_n, end_index - start_index)
            len_of_output = end_index - start_index

        results = {}
        results["array_of_irradiance_field_clearsky"] = [[]] * len_of_output
        results["array_of_pressure"] = [[]] * len_of_output
        results["array_of_altitude"] = [[]] * len_of_output
        results["array_of_latitude"] = [[]] * len_of_output
        results["array_of_longitude"] = [[]] * len_of_output
        results["array_of_index"] = [[]] * len_of_output

        if self.allsky:
            self.ws.scat_dataCalc(interp_order=1)
            self.ws.Delete(self.ws.scat_data_raw)
            self.ws.scat_dataCheck(check_type="all")

            self.ws.dobatch_calc_agenda = fsa.dobatch_calc_agenda_allsky(self.ws)
            self.ws.DOBatchCalc(robust=1)

            temp = np.squeeze(np.array(self.ws.dobatch_irradiance_field.value.copy()))

            results["array_of_irradiance_field_allsky"] = [[]] * len_of_output
            for i in range(len_of_output):
                results["array_of_irradiance_field_allsky"][i] = temp[i, :, :]
            print("...allsky done")

        else:
            self.ws.scat_species = []
            self.ws.scat_data_checked = 1
            self.ws.Touch(self.ws.scat_data)

        self.ws.dobatch_calc_agenda = fsa.dobatch_calc_agenda_clearsky(self.ws)
        self.ws.DOBatchCalc(robust=1)

        temp = np.squeeze(np.array(self.ws.dobatch_irradiance_field.value.copy()))

        for i in range(len_of_output):
            results["array_of_irradiance_field_clearsky"][i] = temp[i, :, :]
            results["array_of_pressure"][i] = (
                atmospheres[i + start_index].grids[1].value[:]
            )
            results["array_of_altitude"][i] = atmospheres[i + start_index].data[
                1, :, 0, 0
            ]
            results["array_of_latitude"][i] = self.ws.matrix_of_Lat.value[
                i + start_index, 0
            ]
            results["array_of_longitude"][i] = self.ws.matrix_of_Lon.value[
                i + start_index, 0
            ]
            results["array_of_index"][i] = i + start_index

        print("...clearsky done")

        return results


# %% addional functions

def generate_gridded_field_from_profiles(pressure_profile,temperature_profile,z_field=None,gases={},particulates={}):
    '''
    Generate a gridded field from profiles of pressure, temperature, altitude, gases and particulates.

    Parameters:
    -----------
    pressure_profile : array
        Pressure profile in Pa.

    temperature_profile : array
        Temperature profile in K.

    z_field : array, optional
        Altitude profile in m. If not provided, it is calculated from the pressure profile.

    gases : dict
        Dictionary with the gas species as keys and the volume mixing ratios as values.

    particulates : dict
        Dictionary with the particulate species with the name of quantity as keys and the quantity values.
        E.g. {'LWC-mass_density': LWC_profile} mass density of liquid water content in kg/m^3.
    Returns:
    --------
    atm_field : GriddedField4
        Gridded field with the profiles of pressure, temperature, altitude, gases and particulates.

        '''

    atm_field=arts.GriddedField4()

    #Do some checks
    if len(pressure_profile) != len(temperature_profile):
        raise ValueError('Pressure and temperature profile must have the same length')

    if z_field is not None and len(pressure_profile) != len(z_field):
        raise ValueError('Pressure and altitude profile must have the same length')

    #Generate altitude field if not provided
    if z_field is None:
        z_field = 16e3 * (5 - np.log10(pressure_profile))

    #set up grids
    abs_species = [f'abs_species-{key}' for key in list(gases.keys())]
    scat_species = [f'scat_species-{key}' for key in list(particulates.keys())]
    atm_field.set_grid(0, ['T','z'] + abs_species + scat_species)
    atm_field.set_grid(1, pressure_profile)

    #set up data
    atm_field.data = np.zeros((len(atm_field.grids[0]),len(atm_field.grids[1]),1,1))

    #The first two values are temperature and altitude
    atm_field.data[0,:,0,0] = temperature_profile
    atm_field.data[1,:,0,0] = z_field

    #The next values are the gas species
    for i,key in enumerate(list(gases.keys())):
        atm_field.data[i+2,:,0,0] = gases[key]


    #The next values are the particulates
    for i,key in enumerate(list(particulates.keys())):
        atm_field.data[i+2+len(gases.keys()),:,0,0] = particulates[key]

    return atm_field
