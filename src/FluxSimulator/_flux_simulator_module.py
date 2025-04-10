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
from copy import deepcopy
from pyarts import cat, xml, arts, version
from pyarts.workspace import Workspace
from . import _flux_simulator_agendas as fsa

# %%


class FluxSimulationConfig:
    """
    This class defines the basic setup for the flux simulator.
    """

    def __init__(self, setup_name, catalog_version=None):
        """
        Parameters
        ----------
        setup_name : str
            Name of the setup. This name is used to create the directory for the LUT.

        Returns
        -------
        None.
        """

        # check version
        version_min = [2, 6, 2]
        v_list = version.split(".")
        major = int(v_list[0]) == version_min[0]
        minor = int(v_list[1]) == version_min[1]
        patch = int(v_list[2]) >= version_min[2]

        if not major or not minor or not patch:
            raise ValueError(
                f"Please use pyarts version >= {'.'.join(str(i) for i in version_min)}."
            )

        self.setup_name = setup_name

        # set default species
        self.species = [
            "H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400",
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
        
        self.LUT_wide_h2o_vmr_default_parameters=[2,-12,-9]

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
        self.quadrature_weights = arts.Vector([])

        # set if allsky or clearsky
        self.allsky = False

        # set if gas scattering is used
        self.gas_scattering = False

        # set default LUT path
        self.lut_path = os.path.join(os.getcwd(), "cache", setup_name)
        self.lutname_fullpath = os.path.join(self.lut_path, "LUT.xml")

        cat.download.retrieve(version=self.catalog_version, verbose=False)

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

    def __init__(self, setup_name, catalog_version=None):
        """
        This class defines the ARTS setup.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """

        super().__init__(setup_name, catalog_version=catalog_version)

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
        self.ws = fsa.set_iy_surface_agendas(self.ws)

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

    def delete_sun(self):
        """
        Delete the sun source from the ARTS WS.

        Returns
        -------
        None.
        """

        try:
            if len(self.ws.suns.value) > 0:
                self.ws.Delete(self.ws.suns)
                self.ws.Touch(self.ws.suns)
                self.ws.suns_do = 0
        except:
            pass

    def set_sun(self, sun_pos=[1.495978707e11, 0.0, 0.0]):
        """
        Sets the sun source for the flux simulator.
        Parameters:
            sun_pos (list, optional): The position of the sun source. Defaults to [1.495978707e11, 0.0, 0.0].
        Returns:
            None
        Raises:
            RuntimeError: If no f_grid is defined.
        Notes:
            - This method deletes any existing sun source before setting the new one.
            - If no sun source is defined, the method sets the suns off.
            - The sun source can be set as a single blackbody or from a gridded field.
            - The sun spectrum can be specified using the sunspectrumtype parameter.
        """

        # delete existing sun
        self.delete_sun()

        try:
            len(self.ws.f_grid.value)

        except RuntimeError:
            print("No f_grid defined!")
            print("Please define a f_grid first!")
            return

        # set sun source
        if len(sun_pos) > 0:
            if self.sunspectrumtype == "Blackbody":
                if len(sun_pos) > 0:
                    self.ws.sunsAddSingleBlackbody(
                        distance=sun_pos[0], latitude=sun_pos[1], longitude=sun_pos[2]
                    )
                else:
                    self.ws.sunsAddSingleBlackbody()

            elif len(self.sunspectrumtype) > 0:
                sunspectrum = arts.GriddedField2()
                if self.sunspectrumtype == "SpectrumMay2004":
                    sunspectrum.readxml("star/Sun/solar_spectrum_May_2004.xml")
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

    def get_sun(self):
        """
        Returns the sun from the ARTS WS.

        Returns:
            float: The value of the first sun.
        """
        try:
            suns = self.ws.suns.value
        except:
            print("No sun variable initialized!")
            suns = None
        return suns

    def scale_sun_to_specific_TSI_at_TOA(self, TSI, latitude, longitude, TOA_altitude):

        try:
            len(self.ws.suns.value)
        except:
            print("No sun source defined!")
            print("Please define a sun source first!")
            return

        r_toa = self.ws.refellipsoid.value[0] + TOA_altitude

        x_toa = r_toa * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
        y_toa = r_toa * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
        z_toa = r_toa * np.sin(np.deg2rad(latitude))

        r_sun = self.ws.suns.value[0].distance
        latitude_sun = self.ws.suns.value[0].latitude
        lomgitude_sun = self.ws.suns.value[0].longitude

        x_sun = (
            r_sun * np.cos(np.deg2rad(latitude_sun)) * np.cos(np.deg2rad(lomgitude_sun))
        )
        y_sun = (
            r_sun * np.cos(np.deg2rad(latitude_sun)) * np.sin(np.deg2rad(lomgitude_sun))
        )
        z_sun = r_sun * np.sin(np.deg2rad(latitude_sun))

        distance = np.sqrt(
            (x_sun - x_toa) ** 2 + (y_sun - y_toa) ** 2 + (z_sun - z_toa) ** 2
        )

        TSI_sun = np.trapz(self.ws.suns.value[0].spectrum[:, 0], self.ws.f_grid.value)

        Radius_sun = self.ws.suns.value[0].radius
        sin_alpha2 = Radius_sun**2 / (distance**2 + Radius_sun**2)

        scale_factor = TSI / TSI_sun / sin_alpha2

        self.ws.suns.value[0].spectrum *= scale_factor

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

    def add_species(self, list_of_species, verbose=False):
        """
        Add new species to the existing species list.

        Parameters:
        - list_of_species (list): A list of species to be added.
        - verbose (bool, optional): If True, print the species that are appended. Default is False.

        Returns:
        None
        """

        # get species list from class NOT from WS
        existing_species = self.species

        new_species = deepcopy(existing_species)

        for spc in list_of_species:
            temp = [j for j, x in enumerate(existing_species) if str(spc) in x]

            if len(temp) == 0:
                new_species.append(str(spc))

                if verbose:
                    print(f"appended: {str(spc)}")

        self.set_species(new_species)

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
                        0.0,
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
        p_min=0.5,
        p_max=1.1e5,
        p_step=0.05,
        lines_speedup_option="None",
        F_grid_from_LUT=False,
        cutoff=True,
        fmin=0,
        fmax=np.inf,
        recalc=False,
        nls_pert=[],
        **kwargs,
    ):
        """
        Generates or retrieves a lookup table based on abs_lookupSetupWide for absorption calculations.
        This method either loads a pre-existing lookup table (LUT) from storage or 
        calculates a new one if the requested one doesn't exist or needs to be recalculated.
        Parameters
        ----------
        t_min : float, optional
            Minimum temperature in Kelvin for the LUT, default 150.0 K
        t_max : float, optional
            Maximum temperature in Kelvin for the LUT, default 350.0 K
        p_min : float, optional
            Minimum pressure in Pa for the LUT, default 0.5 Pa
        p_max : float, optional
            Maximum pressure in Pa for the LUT, default 1.1e5 Pa
        p_step : float, optional
            Pressure step size in logarithmic scale, default 0.05
        lines_speedup_option : str, optional
            Option for line-by-line calculation speedup, default "None"
        F_grid_from_LUT : bool, optional
            Whether to use the frequency grid from the LUT, default False
        cutoff : bool, optional
            Whether to apply a cutoff to absorption lines, default True
        fmin : float, optional
            Minimum frequency to consider in the LUT, default 0
        fmax : float, optional
            Maximum frequency to consider in the LUT, default infinity
        recalc : bool, optional
            Force recalculation of the LUT even if one exists, default False
        nls_pert : list, optional
            Non-LTE perturbations to apply, default empty list
        **kwargs : dict
            Additional keyword arguments passed to abs_lookupSetupWide
        Returns
        -------
        None
            The lookup table is stored internally in the workspace
        Notes
        -----
        - When water vapor is present in abs_species, a default profile is applied based on
          the LUT_wide_h2o_vmr_default_parameters attribute of the class.
        - If recalc is False, the method attempts to read an existing LUT and only recalculates
          if it fails to find one or if it doesn't fit the requested parameters.
        - The generated LUT is stored in the location specified by lutname_fullpath.
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
            try:
                self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
            except RuntimeError:
                print("no lines in abs_species")

            if cutoff == True:
                self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

            if len([ str(tag) for tag in self.get_species().value if "ContCKDMT400" in str(tag)]):
                self.ws.ReadXML(self.ws.predefined_model_data,'model/mt_ckd_4.0/H2O.xml')

            # setup LUT
            print("...setting up lut\n")
            self.ws.abs_lookupSetupWide(t_min=t_min, t_max=t_max, p_step=p_step, p_min=p_min, p_max=p_max, **kwargs)

            # SET abs_vwmrs by hand
            abs_vmrs = self.ws.abs_vmrs.value

            # check for water vapor
            H2O_exist = [
                True if "H2O" in str(x) else False for x in self.ws.abs_species.value
            ]

            if np.sum(H2O_exist):

                vmr_h20_default_profile = self.ws.abs_p.value[:] ** (
                    self.LUT_wide_h2o_vmr_default_parameters[0]
                ) * 10 ** (self.LUT_wide_h2o_vmr_default_parameters[1])

                vmr_h20_default_profile[
                    vmr_h20_default_profile
                    < self.LUT_wide_h2o_vmr_default_parameters[2]
                ] = 2

                for i, logic in enumerate(H2O_exist):
                    if logic:
                        abs_vmrs[i, :] = vmr_h20_default_profile

            self.ws.abs_vmrs.value = abs_vmrs

            # add different nls_pert
            if len(nls_pert) > 0:
                self.ws.abs_nls_pert = nls_pert

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
        atm,
        p_step=0.05,
        lines_speedup_option="None",
        F_grid_from_LUT=False,
        cutoff=True,
        fmin=0,
        fmax=np.inf,
        recalc=False,
        **kwargs,
    ):
        """
        Creates or retrieves a look-up table (LUT) based on abs_lookupSetup for absorption calculations.
        This method either loads an existing LUT or calculates a new one based on the given atmosphere.
        The LUT is used to speed up absorption calculations in radiative transfer simulations.
        Parameters
        ----------
        atm : object
            Atmospheric state object containing temperature, pressure, and species concentrations.
        p_step : float, optional
            Pressure grid step size for the LUT, default is 0.05.
        lines_speedup_option : str, optional
            Option for line absorption calculation speedup, default is "None".
        F_grid_from_LUT : bool, optional
            Whether to use frequency grid from the LUT, default is False.
        cutoff : bool, optional
            Whether to apply cutoff to absorption lines, default is True.
        fmin : float, optional
            Minimum frequency to consider in Hz, default is 0.
        fmax : float, optional
            Maximum frequency to consider in Hz, default is infinity.
        recalc : bool, optional
            Force recalculation of the LUT even if it exists, default is False.
        **kwargs : dict
            Additional keyword arguments passed to abs_lookupSetup.
        Returns
        -------
        None
            The look-up table is stored internally and can be accessed via self.ws.abs_lookup.
        Notes
        -----
        If cutoff is True, absorption lines beyond 750 GHz are ignored. The method automatically
        handles the MT_CKD_4.0 continuum model if it's included in the species list.
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
            try:
                self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
            except RuntimeError:
                print("no lines in abs_species")

            if cutoff == True:
                self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

            if len([ str(tag) for tag in self.get_species().value if "ContCKDMT400" in str(tag)]):
                self.ws.ReadXML(self.ws.predefined_model_data,'model/mt_ckd_4.0/H2O.xml')

            self.ws.atm_fields_compact = atm
            self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

            # setup LUT
            print("...setting up lut\n")
            self.ws.atmfields_checked = 1
            self.ws.abs_lookupSetup(p_step=p_step, **kwargs)

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

    def get_lookuptable_profile(
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
        Generate or retrieve a lookup table (LUT) for atmospheric absorption profiles.
        This method creates a lookup table for the radiative transfer calculations
        based on given atmospheric profiles or loads a previously calculated LUT if available.
        Parameters
        ----------
        pressure_profile : array-like
            Pressure levels of the atmosphere [Pa].
        temperature_profile : array-like
            Temperature profile at each pressure level [K].
        vmr_profiles : array-like
            Volume mixing ratios for each species at each pressure level.
            Shape should be (n_species, n_pressure_levels).
        p_step : float, optional
            Pressure grid step size for the lookup table. Default is 0.05.
        lines_speedup_option : str, optional
            Speedup option for line calculations. Default is "None".
        F_grid_from_LUT : bool, optional
            Whether to use the frequency grid from the LUT. Default is False.
        cutoff : bool, optional
            Whether to apply line cutoff at 750 GHz. Default is True.
        fmin : float, optional
            Minimum frequency for LUT calculations [Hz]. Default is 0.
        fmax : float, optional
            Maximum frequency for LUT calculations [Hz]. Default is infinity.
        recalc : bool, optional
            Force recalculation of the LUT even if it exists. Default is False.
        Returns
        -------
        None
            The LUT is stored in the workspace and saved to disk.
        Raises
        ------
        ValueError
            If the dimensions of pressure_profile, temperature_profile, and vmr_profiles are inconsistent.
        RuntimeError
            If reading the existing LUT fails.
        Notes
        -----
        The method performs the following steps:
        1. Tries to read an existing LUT if recalc=False
        2. If reading fails or recalc=True, it sets up and calculates a new LUT
        3. Validates profiles dimensions
        4. Sets up the ARTS workspace with the atmospheric data
        5. Calculates the absorption lookup table
        6. Saves the LUT to disk
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
                self.ws.abs_lines_per_speciesNormalization(option="SFS")

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

    def get_lookuptableBatch(
        self,
        batch_atmospheres,
        p_step=0.05,
        lines_speedup_option="None",
        F_grid_from_LUT=False,
        cutoff=True,
        fmin=0,
        fmax=np.inf,
        recalc=False,
    ):
        """
        This function calculates the LUT using the batch setup.
        It inputs a batch of atmospheres and calculates the Lut for this batch.


        Parameters
        ----------
        batch_atmospheres : ArrayOfGriddedField4
            Batch of atmospheres.
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

            # set atmosphere
            self.ws.batch_atm_fields_compact = batch_atmospheres

            if cutoff == True:
                self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

            # setup LUT
            print("...setting up lut\n")
            self.ws.abs_lookupSetupBatch(p_step=p_step)

            # SET abs_vwmrs by hand
            abs_vmrs=self.ws.abs_vmrs.value           

            # check for water vapor
            H2O_exist=[True  if "H2O" in str(x) else False for x in self.ws.abs_species.value]
            # H2Osum=np.sum(H2O_exist)

            # Here we modify the vmr of H2O if there are more than one H2O species,
            # because in ARTS only the first one is set correctly (You can call this buggy behavior).
            # So, we set it to same value as the first H2O species.

            if np.sum(H2O_exist):         
                flag=False
                for i, logic in enumerate(H2O_exist):

                    if logic:
                        if flag==False:
                            vmrH2O_min=np.min(abs_vmrs[i,:])
                            flag=True

                        logic2=abs_vmrs[i,:]==0
                        abs_vmrs[i,logic2]=vmrH2O_min

                self.ws.abs_vmrs.value=abs_vmrs

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
        **kwargs,
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
            Geographical position of the simulated atmosphere.
            Needs to be set for solar simulations.

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

        # prepare atmosphere
        self.ws.atm_fields_compact = atm
        self.check_species()
        self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

        # set absorption
        # =============================================================================

        print("setting up absorption...\n")

        # Calculate or load LUT
        self.get_lookuptableWide(**kwargs)

        # Use LUT for absorption
        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        # setup
        # =============================================================================

        # surface altitudes
        self.ws.z_surface = [[z_surface]]

        # surface temperatures
        self.ws.surface_skin_t = T_surface

        # set geographical position
        if len(geographical_position) == 0:
            self.ws.lat_true = [0]
            self.ws.lon_true = [0]

            if self.ws.suns_do == 1:
                raise ValueError(
                    "You have defined a sun source but no geographical position!\n"
                    + "Please define a geographical position!"
                    + "The position is needed to calculate the solar zenith angle."
                    + "Thanks!"
                )

        else:
            self.ws.lat_true = [geographical_position[0]]
            self.ws.lon_true = [geographical_position[1]]

        # surface reflectivities
        try:            
            self.ws.surface_scalar_reflectivity = surface_reflectivity
        except:            
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
            self.ws.gas_scattering_do = 1

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

        aux_var_allsky = []
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
                    aux_var_allsky.append(self.ws.disort_aux.value[i][:] * 1.0)

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
        aux_var_clearsky = []
        if len(self.ws.disort_aux_vars.value):
            for i in range(len(self.ws.disort_aux_vars.value)):
                aux_var_clearsky.append(self.ws.disort_aux.value[i][:] * 1.0)

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

    def flux_simulator_single_profile_NoLut(
        self,
        atm,
        T_surface,
        z_surface,
        surface_reflectivity,
        geographical_position=np.array([]),
        cutoff=True,
        lines_speedup_option="None",
    ):
        """
        Perform a single atmospheric radiative flux calculation without using a Look-Up Table (NoLUT).
        This method computes radiative fluxes and heating rates for a given atmospheric profile.
        It can calculate both clearsky and all-sky (with clouds/particles) conditions depending on
        the configuration of the FluxSimulator instance.
        Parameters
        ----------
         atm : GriddedField4 
            Atmospheric profile data in ARTS compact format.
        T_surface : float
            Surface temperature in Kelvin.
        z_surface : float
            Surface altitude in meters.
        surface_reflectivity : float or array-like
            Surface reflectivity value(s).
        geographical_position : np.ndarray, optional
            Array containing [latitude, longitude] in degrees.
            Default is an empty array, which sets position to [0, 0].
            Required if solar radiation (suns_do=1) is included.
        cutoff : bool, optional
            Whether to apply a frequency cutoff to the absorption lines.
            Default is True.
        lines_speedup_option : str, optional
            Option for line-by-line calculation speedup.
            Default is "None".
        Returns
        -------
        dict
            Dictionary containing:
            - flux_clearsky_up : Upward clearsky integrated flux
            - flux_clearsky_down : Downward clearsky integrated flux
            - spectral_flux_clearsky_up : Spectral upward clearsky flux
            - spectral_flux_clearsky_down : Spectral downward clearsky flux
            - heating_rate_clearsky : Clearsky heating rate in K/day
            - pressure : Pressure grid
            - altitude : Altitude grid
            - f_grid : Frequency grid
            - aux_var_clearsky : Auxiliary variables for clearsky calculation
            If all-sky calculations are enabled, also includes:
            - flux_allsky_up : Upward all-sky integrated flux
            - flux_allsky_down : Downward all-sky integrated flux
            - spectral_flux_allsky_up : Spectral upward all-sky flux
            - spectral_flux_allsky_down : Spectral downward all-sky flux
            - heating_rate_allsky : All-sky heating rate in K/day
            - aux_var_allsky : Auxiliary variables for all-sky calculation
        Notes
        -----
        This method uses DISORT for the radiative transfer calculation and 
        performs on-the-fly absorption calculations without relying on pre-computed
        look-up tables.
        """
        

        # define environment
        # =============================================================================

        # prepare atmosphere
        self.ws.atm_fields_compact = atm
        self.check_species()
        self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

        # set absorption
        # =============================================================================

        print("setting up absorption...\n")

        # read spectroscopic data
        print("...reading data\n")
        self.ws.ReadXsecData(basename="xsec/")
        self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")

        if cutoff == True:
            self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

        # Use on the fly absorption
        self.ws.propmat_clearsky_agendaAuto(lines_speedup_option=lines_speedup_option)

        if cutoff == True:
            self.ws.lbl_checked = 1
        else:
            self.ws.lbl_checkedCalc()

        # setup
        # =============================================================================

        # surface altitudes
        self.ws.z_surface = [[z_surface]]

        # surface temperatures
        self.ws.surface_skin_t = T_surface

        # set geographical position
        if len(geographical_position) == 0:
            self.ws.lat_true = [0]
            self.ws.lon_true = [0]

            if self.ws.suns_do == 1:
                raise ValueError(
                    "You have defined a sun source but no geographical position!\n"
                    + "Please define a geographical position!"
                    + "The position is needed to calculate the solar zenith angle."
                    + "Thanks!"
                )

        else:
            self.ws.lat_true = [geographical_position[0]]
            self.ws.lon_true = [geographical_position[1]]

        # surface reflectivities
        try:            
            self.ws.surface_scalar_reflectivity = surface_reflectivity
        except:            
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
            self.ws.gas_scattering_do = 1

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

        aux_var_allsky = []
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
                    aux_var_allsky.append(self.ws.disort_aux.value[i][:] * 1.0)

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
        aux_var_clearsky = []
        if len(self.ws.disort_aux_vars.value):
            for i in range(len(self.ws.disort_aux_vars.value)):
                aux_var_clearsky.append(self.ws.disort_aux.value[i][:] * 1.0)

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

        # set sun
        self.set_sun()
        self.ws.IndexCreate("sun_index")
        self.ws.sun_index = 0
        if len(self.get_sun()) == 0:
            self.ws.sun_index = -999

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
            self.ws.gas_scattering_do = 1

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

    def calc_optical_thickness(
        self,
        atm,
        T_surface,
        z_surface,
        surface_reflectivity,
        geographical_position=np.array([]),
        **kwargs,
    ):

        # define environment
        # =============================================================================

        # prepare atmosphere
        self.ws.atm_fields_compact = atm
        self.check_species()
        self.ws.AtmFieldsAndParticleBulkPropFieldFromCompact()

        # set absorption
        # =============================================================================

        print("setting up absorption...\n")

        # Calculate or load LUT
        self.get_lookuptableWide(**kwargs)

        # Use LUT for absorption
        self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)

        # setup
        # =============================================================================

        # surface altitudes
        self.ws.z_surface = [[z_surface]]

        # surface temperatures
        self.ws.surface_skin_t = T_surface

        # set geographical position
        if len(geographical_position) == 0:
            self.ws.lat_true = [0]
            self.ws.lon_true = [0]

            if self.ws.suns_do == 1:
                raise ValueError(
                    "You have defined a sun source but no geographical position!\n"
                    + "Please define a geographical position!"
                    + "The position is needed to calculate the solar zenith angle."
                    + "Thanks!"
                )

        else:
            self.ws.lat_true = [geographical_position[0]]
            self.ws.lon_true = [geographical_position[1]]

        # surface reflectivities
        try:            
            self.ws.surface_scalar_reflectivity = surface_reflectivity
        except:            
            self.ws.surface_scalar_reflectivity = [surface_reflectivity]   

        print("starting calculation...\n")

        # no sensor
        self.ws.sensorOff()

        self.ws.atmfields_checkedCalc()

        self.ws.propmat_clearsky_fieldCalc()

        abs_coeff=self.ws.propmat_clearsky_field.value[:,:,0,0,:,0,0]

        # Calculate optical thickness
        z_field = atm.data[1, :, 0, 0]
        dz = np.diff(z_field)
        dz = np.append(np.array(dz[-1]),dz )

        optical_thickness = np.cumsum(abs_coeff[:,:,::-1]*dz[::-1], axis=2)
        optical_thickness = optical_thickness[:,:,::-1]

        # results
        # ====================================================================================

        return optical_thickness


# %% addional functions


def generate_gridded_field_from_profiles(
    pressure_profile, temperature_profile, z_field=None, gases={}, particulates={}
):
    """
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

    """

    atm_field = arts.GriddedField4()

    # Do some checks
    if len(pressure_profile) != len(temperature_profile):
        raise ValueError("Pressure and temperature profile must have the same length")

    if z_field is not None and len(pressure_profile) != len(z_field):
        raise ValueError("Pressure and altitude profile must have the same length")

    # Generate altitude field if not provided
    if z_field is None:
        z_field = 16e3 * (5 - np.log10(pressure_profile))

    # set up grids
    abs_species = [f"abs_species-{key}" for key in list(gases.keys())]
    scat_species = [f"scat_species-{key}" for key in list(particulates.keys())]
    atm_field.set_grid(0, ["T", "z"] + abs_species + scat_species)
    atm_field.set_grid(1, pressure_profile)

    # set up data
    atm_field.data = np.zeros((len(atm_field.grids[0]), len(atm_field.grids[1]), 1, 1))

    # The first two values are temperature and altitude
    atm_field.data[0, :, 0, 0] = temperature_profile
    atm_field.data[1, :, 0, 0] = z_field

    # The next values are the gas species
    for i, key in enumerate(list(gases.keys())):
        atm_field.data[i + 2, :, 0, 0] = gases[key]

    # The next values are the particulates
    for i, key in enumerate(list(particulates.keys())):
        atm_field.data[i + 2 + len(gases.keys()), :, 0, 0] = particulates[key]

    return atm_field
