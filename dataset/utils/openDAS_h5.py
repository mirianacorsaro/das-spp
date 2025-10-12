import h5py
import numpy as np

def read_febus(pathname, giunzioni_path = 'data/RAWDATA/info/Giunzioni.txt'):

    try:
        with h5py.File(pathname, 'r') as file:
            das_name = list(file.keys())[0]
            zone_to_plot = 1

            version = file[das_name + '/Source1/'].attrs['Version']
            acquisition_length = file[das_name + '/Source1/'].attrs['FiberLength']
            samp = file[das_name + '/Source1/'].attrs['SamplingRate']
            spacing = file[das_name + f'/Source1/Zone{zone_to_plot}/'].attrs['Spacing']
            sampling_res = spacing[0] * 1e2
            origin = file[das_name + f'/Source1/Zone{zone_to_plot}/'].attrs['Origin']
            extent = file[das_name + f'/Source1/Zone{zone_to_plot}/'].attrs['Extent']
            block_overlap = file[das_name + f'/Source1/Zone{zone_to_plot}/'].attrs['BlockOverlap']
            
            zi_start = origin[0] + (extent[0]) * spacing[0]
            zi_end = origin[0] + (extent[1]) * spacing[0]
            distance_fiber = np.arange(zi_start, zi_end + spacing[0], spacing[0])
            dt = spacing[1] * 1e-3
            
            path1 = das_name + f'/Source1/Zone{zone_to_plot}/Strain Rate [nStrain|s]'
            path2 = das_name + f'/Source1/Zone{zone_to_plot}/StrainRate'

            if path1 in file:
                strain_rate_dataset = file[path1]
            elif path2 in file:
                strain_rate_dataset = file[path2]
            else:
                raise KeyError(f"Neither '{path1}' nor '{path2}' exist in the file.")

            data_hdf5_rh5_s = np.array(strain_rate_dataset)
            dims = data_hdf5_rh5_s.shape
            overlap = int(dims[1] - dims[1] / (1 + block_overlap / 100))
            block = int(dims[1] / (1 + block_overlap / 100))

            time_dataset = file[das_name + '/Source1/time']
            time = np.array(time_dataset)
            
        t_set = time - block / 2 * (spacing[1] * 1e-3)
        strain_rate_d = np.reshape(data_hdf5_rh5_s[:, overlap//2:-overlap//2, :], (block*dims[0], dims[2])).T

        total_time_size = int(block * dims[0])
        total_time = np.arange(0, total_time_size*spacing[1] * 1e-3, spacing[1] * 1e-3)

        with open(giunzioni_path) as f:
            lines = f.readlines()
            
            for c in lines:
                first_ch, last_ch = int(c.split(' ')[0])-1, int(c.split(' ')[1].split('\n')[0])-1
                strain_rate_d = np.delete(strain_rate_d, range(first_ch, last_ch), 0)   

        return strain_rate_d

    except OSError as e:
        print(f"Error opening file: {e}")

        return []
    
