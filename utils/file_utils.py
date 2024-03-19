import pickle
import h5py

import numpy as np

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def print_h5_file_info(h5_file_path):
    with h5py.File(h5_file_path, "r") as f:

        print("File attributes:", f.attrs.keys())
        print("File keys:", f.keys())

        for i, big_key in enumerate(f.keys()):
            print(f"Key {i}: <{big_key}>")
            dset = f[big_key]
            print(f"Dataset <{big_key}> length:", len(dset))
            print(f"Dataset <{big_key}> shape:", dset.shape)

            print(f"Dataset <{big_key}> attribute keys:", dset.attrs.keys())
            print()

            for key, value in dset.attrs.items():
                print("\t", key, value)

            print()
            for i in range(10):
                if big_key == "imgs":
                    print(dset[i].shape)
                else:
                    print(dset[i].shape, dset[i])
            print()


def compare_attributes(obj1, obj2, obj_name, ignore_attributes):
    if not set(obj1.attrs.keys()) == set(obj2.attrs.keys()):
        print(
            f"Attribute keys of <{obj_name}> in file1 {obj1.attrs.keys()} are not the same as in file2 {obj2.attrs.keys()}")
        return False
    for key in obj1.attrs.keys():
        if not np.array_equal(obj1.attrs[key], obj2.attrs[key]):
            print(f"Attributes for key <{key}> in {obj_name} not equal:")
            print("\t", obj1.attrs[key])
            print("\t", obj2.attrs[key])
            if key in ignore_attributes:
                print(f"Attribute key <{key}> inconsistency is ignored.")
                continue
            else:
                return False
    return True


def check_h5_files_are_equal(file1, file2, atol, ignore_attributes=()):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:

        # Check file attribute keys
        if not compare_attributes(f1, f2, "file", ignore_attributes):
            return False

        # Check dataset keys
        if not set(f1.keys()) == set(f2.keys()):
            print(
            	f"Dataset keys of file1 {f1.keys()} are not the same as dataset keys of file2 {f2.keys()}")
            return False

        # Check datasets and their attributes
        for key in f1.keys():
            dset1 = f1[key]
            dset2 = f2[key]

            # Check dataset attributes
            if not compare_attributes(dset1, dset2, f"dataset <{key}>", ignore_attributes):
                return False

            # Check dataset values
            if not np.allclose(dset1[:], dset2[:], atol=atol):
                print(f"Datasets for key {key} not equal")
                print(dset1[:] - dset2[:])
                return False

    print("The files are the same.")
    return True
