import numpy as np
import scipy as sp
import os, sys
import pandas as pd
import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod

class frezchem_seawater():

    def __init__(self):
        self.phreeqc = phreeqc_mod.IPhreeqc('/usr/local/lib/libiphreeqc.so')
        self.phreeqc.load_database("/home/adminix/iphreeqc-3.8.6-17100/database/frezchem.dat")
        if (self.phreeqc.phc_database_error_count > 0):
            raise RuntimeError(f"phreeqc database could not be loaded: {self.phreeqc.phc_database_error_count} errors.")
        
        self.ref_molmasses = np.array([22.98977, 35.453, 24.305, 96.0626, 40.078, 39.0983]) #g/mol
        self.ref_solutes = np.array([0.468, 0.545, 0.052, 0.028, 0.010, 0.010])  #(mol/kgw) @ ~35.2 g/kg salinity - adds up to 34.8265 g/kg

    def solve_at(self, tempC, salinity, logError=False):
        iceFrac, gasSat = 0, 0
        try:
            self.phreeqc.run_string(self.single_script(tempC, salinity))
            output = self.phreeqc.get_selected_output_array()
            tmp1 = output[0].index('m_H2O(mol/kgw)')
            tmp2 = np.array([output[2][i] for i in range(len(output[0])) if ('(mol/kgw)' in output[0][i] and i != tmp1)])
            iceFrac = output[2][output[0].index('Ice(s)')] / output[2][tmp1]
            #gasFrac = ???
        except:
            if (logError):
                print('failure')
                iceFrac = 1
            return (iceFrac, gasSat)
        
        return iceFrac, gasSat#pd.DataFrame(output[1:], columns=output[0]).iloc[:,5:]
    
    def single_script(self, tempC, salinity):
        solutes = self.ref_solutes * (salinity / 35.2)
        working_script = f"""
        SOLUTION 1 Seawater-like composition
            temp      {tempC}      # Starting temperature in °C (to be changed each run)
            pH        8.0
            pressure  1
            water     1  # kg
            units     mol/kgw
            Na        {solutes[0]} #22.98977 g/mol
            Cl        {solutes[1]} #35.453
            Mg        {solutes[2]} #24.305
            SO4       {solutes[3]} #96.0626
            Ca        {solutes[4]} #40.078
            K         {solutes[5]} #39.0983
        """
        working_script += """
        EQUILIBRIUM_PHASES 1
        
        # Phase name	# Target SI	# mol	# precipitate_only	# Formula
            Ice(s)		0		0				# H2O
        
        # Chlorides
            Halite		0		0				# NaCl
            Hydrohalite	0		0				# NaCl:2H2O
            Sylvite		0		0				# KCl
            Antarcticite	0		0				# CaCl2:6H2O
            Bischofite	0		0				# MgCl2:6H2O
            MgCl2:8H2O	0		0				# MgCl2:8H2O
            MgCl2:12H2O	0		0				# MgCl2:12H2O
            Carnallite	0		0				# KMgCl3:6H2O
        
        # Sulfates
            Anhydrite	0		0				# CaSO4	
            Arcanite	0		0				# K2SO4
            Epsomite	0		0				# MgSO4:7H2O
            Gypsum		0		0				# CaSO4:2H2O
            Mirabilite	0		0				# Na2SO4:10H2O
            Picromerite	0		0				# MgSO4:K2SO4:6H2O
        """
        working_script += """
        SELECTED_OUTPUT
            -temperature true
            -pH          true
            -molalities  H2O
            -equilibrium_phases Ice(s)
        """
        return working_script