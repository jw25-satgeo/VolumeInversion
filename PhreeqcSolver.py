import numpy as np
import scipy
import os, sys
import pandas as pd
import phreeqpy.iphreeqc.phreeqc_dll as phreeqc_mod
import concurrent.futures

class frezchem_seawater():

    def __init__(self):
        self.ref_sollabs = ['Na', 'Cl', 'Mg', 'S', 'Ca', 'K']
        self.ref_molmasses = np.array([22.98977, 35.453, 24.305, 32.07, 40.078, 39.0983]) #g/mol
        self.ref_solutes = np.array([0.468, 0.545, 0.052, 0.028, 0.010, 0.010])  #(mol/kgw) @ ~35.2 g/kg salinity - adds up to 34.8265 g/kg

    def create_table(self, TminC=-30, TmaxC=20, Tc=100, Smin=0.1, Smax=300.1, Sc=100, logError=False):
        temps, salis = np.linspace(TminC, TmaxC, Tc), np.linspace(Smin, Smax, Sc)
        temps, salis = np.meshgrid(temps, salis, indexing='ij')
        
        results = np.zeros((Tc, Sc, 2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.solve_at, temps[i,j], salis[i,j], (i,j), logError) for j in range(Sc) for i in range(Tc)]
            for future in concurrent.futures.as_completed(futures):
                iceFrac, brinSal, tag = future.result()
                results[tag[0],tag[1],0] = iceFrac
                results[tag[0],tag[1],1] = brinSal

        # fill NaN values
        coords = np.meshgrid(np.arange(results.shape[0]),np.arange(results.shape[1]),indexing='ij') #(Tc, Sc), (Tc, Sc)
        mask = np.isnan(results).any(axis=-1) # Tc, Sc
        if (np.any(mask)):
            fill_results = scipy.interpolate.RBFInterpolator(
                np.stack((coords[0][~mask], coords[1][~mask]), axis=-1),
                np.stack((results[:,:,0][~mask], results[:,:,1][~mask]), axis=-1),
                smoothing=1e-6
            )(np.stack((coords[0][mask], coords[1][mask]), axis=-1))
            results[...,0][mask] = np.clip(fill_results[...,0], 0, 1)
            results[...,1][mask] = np.clip(fill_results[...,1], 0, None)
        
        return results
    
    def solve_at(self, tempC, salinity, tag=None, logError=False):
        iceFrac, gasSat = 0, 0
        thread_phreeqc = phreeqc_mod.IPhreeqc('/usr/local/lib/libiphreeqc.so')
        thread_phreeqc.load_database("/home/adminix/iphreeqc-3.8.6-17100/database/frezchem.dat")
        if (thread_phreeqc.phc_database_error_count > 0):
            raise RuntimeError(f"phreeqc database could not be loaded: {thread_phreeqc.phc_database_error_count} errors.")
        try:
            thread_phreeqc.run_string(self.single_script(tempC, salinity))
            output = thread_phreeqc.get_selected_output_array()
            #return output
            tmp1 = output[0].index('m_H2O(mol/kgw)')
            tmp2 = np.array([output[-1][i] for i in range(len(output[0])) if ('(mol/kgw)' in output[0][i] and i != tmp1)])
            iceFrac = output[-1][output[0].index('Ice(s)')] / output[-1][tmp1]
            brinSal = np.sum(tmp2 * self.ref_molmasses) * 35/32.846983315280845  #mol/kgw * g/mol should equal g/kgw, plus correction factor
            
            #gasFrac = ???
        except:
            if (logError):
                print('failure')
                iceFrac = 1
            return (np.nan, np.nan, tag)
        
        return iceFrac, brinSal, tag

    def solve_at_raw(self, tempC, salinity, tag=None, logError=False):
        thread_phreeqc = phreeqc_mod.IPhreeqc('/usr/local/lib/libiphreeqc.so')
        thread_phreeqc.load_database("/home/adminix/iphreeqc-3.8.6-17100/database/frezchem.dat")
        if (thread_phreeqc.phc_database_error_count > 0):
            raise RuntimeError(f"phreeqc database could not be loaded: {thread_phreeqc.phc_database_error_count} errors.")
        thread_phreeqc.run_string(self.single_script(tempC, salinity))
        output = thread_phreeqc.get_selected_output_array()
        return output
    
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
            S         {solutes[3]} #32.07
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
            -totals      Na Cl Mg S Ca K
            -molalities  H2O
            -equilibrium_phases Ice(s)
        """
        #Halite Hydrohalite Sylvite Antarcticite Bischofite MgCl2:8H2O MgCl2:12H2O Carnallite Anhydrite Arcanite Epsomite Gypsum Mirabilite Picromerite
        return working_script