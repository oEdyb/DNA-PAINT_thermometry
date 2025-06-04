# -*- coding: utf-8 -*-
"""
Created on Fri June 25 13:05:12 2021

@author: Mariano Barella

This script reads and saves the output of Picasso software. Precisely, it opens
hdf5 files that are the output of Picasso's Localize module, Filter module or Render 
module (if data is being filtered or picked) and saves some of the data in a 
.dat file with ASCII encoding. It creates a folder called "split_data".

"""
# ========================== IMPORT LIBRARIES ==========================
import pickle
import numpy as np
import h5py
import os
import re
import tkinter as tk
import tkinter.filedialog as fd

# ========================== MAIN FUNCTION TO SPLIT HDF5 FILES ==========================
def split_hdf5(hdf5_file, folder, recursive_flag, rectangles_flag, lpx_filter,
               lpy_filter, verbose_flag, NP_flag):
    
    print('\nStarting STEP 1.')
    # set directories
    folder = os.path.dirname(hdf5_file)
    video_name = os.path.basename(hdf5_file)
    
    # ================ DETERMINE FILES TO PROCESS BASED ON FLAGS ================
    if recursive_flag:
        list_of_files = os.listdir(folder)
        list_of_files = [f for f in list_of_files if re.search('.hdf5',f)]
        list_of_files.sort()
    elif NP_flag:
        list_of_files = os.listdir(folder)
        list_of_files = [f for f in list_of_files if re.search('NP_subtracted', f)]
        list_of_files = [f for f in list_of_files if re.search('picked.hdf5', f)]
        if len(list_of_files) == 2:
            list_of_files = [list_of_files[0]]
        list_of_files.append(video_name)
    else:
        list_of_files = [video_name]
        
    # ================ PROCESS EACH FILE IN THE LIST ================
    for filename in list_of_files:
        filepath = os.path.join(folder, filename)
        print('\nFile selected:', filepath)
        
        # ================ OPEN AND READ HDF5 FILE ================
        with h5py.File(filepath, 'r') as f:
            # List all groups
            # print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
        
            # Get the data
            data = list(f[a_group_key])

        # ================ ALLOCATE ARRAYS FOR DATA EXTRACTION ================
        frame = np.zeros([len(data)])
        x = np.zeros([len(data)])
        y = np.zeros([len(data)])
        photons = np.zeros([len(data)])
        sx = np.zeros([len(data)])
        sy = np.zeros([len(data)])
        bg = np.zeros([len(data)])
        lpx = np.zeros([len(data)])
        lpy = np.zeros([len(data)])
        ellipticity = np.zeros([len(data)])
        net_gradient = np.zeros([len(data)])
        group = np.zeros([len(data)])

        # ================ EXTRACT DATA FROM HDF5 ================
        for i in range(len(data)):
            frame[i] = data[i][0]
            x[i] = data[i][1]
            y[i] = data[i][2]
            photons[i] = data[i][3]
            sx[i] = data[i][4]
            sy[i] = data[i][5]
            bg[i] = data[i][6]
            lpx[i] = data[i][7]
            lpy[i] = data[i][8]
            ellipticity[i] = data[i][9]
            net_gradient[i] = data[i][10]
            if rectangles_flag:
                group[i] = data[i][13] # if picks are rectangles
            else:
                group[i] = data[i][11] # if picks are circles



        # ================ PREPARE OUTPUT DIRECTORY ================
        save_folder = os.path.join(folder, 'split_data')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # create dictionary to link files
        link_files_dict = {}
        
        clean_filename = filename[:-5]
        clean_filename = clean_filename.replace('MMStack_Pos0.ome_', '')

        # ================ FILTER DATA BY LPX AND LPY VALUES ================
        lpx_filter_low, lpx_filter_high = (0.005, lpx_filter)
        lpy_filter_low, lpy_filter_high = (0.005, lpy_filter)
        # lpx_filter_low, lpx_filter_high = (0, 99)
        # lpy_filter_low, lpy_filter_high = (0, 99)
        filter_index = np.where((lpx > lpx_filter_low) & (lpy > lpy_filter_low) & (lpx < lpx_filter_high)
                                & (lpy < lpy_filter_high) & (net_gradient > 800))


        if NP_flag:
            if 'NP' in filename:
                clean_filename = 'NP_subtracted'
            else:
                clean_filename = 'raw'
        else:
            clean_filename = ''
            
        # ================ SAVE FILTERED DATA TO SEPARATE FILES ================
        # locs
        data_to_save = frame[filter_index]
        new_filename = clean_filename + '_frame.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%i')
        link_files_dict['frame'] = new_filename
        
        # positions
        data_to_save = np.asarray([x[filter_index], y[filter_index]]).T
        new_filename = clean_filename + '_xy.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%.3f')
        link_files_dict['positions'] = new_filename

        # sx, sy
        data_to_save = np.asarray([sx[filter_index], sy[filter_index]]).T
        new_filename = clean_filename + '_sxy.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%.3f')
        link_files_dict['std'] = new_filename

        # lpx, lpy
        data_to_save = np.asarray([lpx[filter_index], lpy[filter_index]]).T
        new_filename = clean_filename + '_lpxy.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%.3f')
        link_files_dict['lp'] = new_filename

        # ellipticity
        data_to_save = ellipticity[filter_index]
        new_filename = clean_filename + '_ellipticity.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%.1f')
        link_files_dict['ellipticity'] = new_filename

        # net gradient
        data_to_save = net_gradient[filter_index]
        new_filename = clean_filename + '_netgrad.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%.1f')
        link_files_dict['netgrad'] = new_filename

        # photons
        data_to_save = photons[filter_index]
        new_filename = clean_filename + '_photons.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%.1f')
        link_files_dict['photons'] = new_filename

        # background
        data_to_save = bg[filter_index]
        new_filename = clean_filename + '_bkg.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%.1f')
        link_files_dict['bkg'] = new_filename
        
        # pick number
        data_to_save = group[filter_index]
        new_filename = clean_filename + '_pick_number.dat'
        new_filepath = os.path.join(save_folder, new_filename)
        np.savetxt(new_filepath, data_to_save, fmt='%i')
        link_files_dict['pick_number'] = new_filename

        # ================ SAVE DICTIONARY WITH LINKS TO FILES ================
        data_to_save = link_files_dict
        new_filename = clean_filename + '_dict.pkl'
        new_filepath = os.path.join(save_folder, new_filename)
        with open(new_filepath, 'wb') as f:
            pickle.dump(data_to_save, f)

    # ================ FINAL SUMMARY OUTPUT ================
    pick_numbers = np.unique(group[filter_index])
    print('\n' + '='*23 + '🔬 STEP 1 SUMMARY 🔬' + '='*23)
    print(f'   Total Files Processed: {len(list_of_files)}')
    print(f'   Picks Found: {len(pick_numbers)}')
    print(f'   Localizations Before LP Filter: {len(frame)}')
    print(f'   Localizations After LP Filter: {len(frame[filter_index])}')
    print(f'   All data saved in "split_data" directory.')
    print('='*70)
    print('\nDone with STEP 1.')
    return

#####################################################################
#####################################################################
#####################################################################

# ========================== SCRIPT EXECUTION BLOCK ==========================
if __name__ == '__main__':
    
    # if picks are rectangles set TRUE
    rectangles_flag = False
    
    # set TRUE to run the script for all the files inside the selected folder
    recursive_flag = False
    
    # load and open folder and file
    base_folder = 'C:\\datos_mariano\\posdoc\\unifr\\DNA-PAINT_nanothermometry\\data_fribourg'
    root = tk.Tk()
    hdf5_file = fd.askopenfilename(initialdir = base_folder, \
                                       filetypes=(("", "*.hdf5"), ("", "*.")))
    root.withdraw()
    
    split_hdf5(hdf5_file, base_folder, recursive_flag, rectangles_flag,
               0.15, 0.15, True)