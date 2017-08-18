"""
A script for building the materials database from the
USGS Spectral Library (https://www.sciencebase.gov/catalog/item/586e8c88e4b0f5ce109fccae)
"""
import wget
import zipfile
import os
import shutil
import pandas as pd
import pickle
import argparse
import sys

from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor

DATA_DIR = 'cycsat/data/'
DATASET = DATA_DIR + 'ASCIIdata_splib07a/'

if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)


def clear_datadir(DATA_DIR):
    """Clears all data from the data directory."""
    print('Clearing data')
    contents = os.listdir(DATA_DIR)
    for path in contents:
        if path == 'info.md':
            continue

        if os.path.isdir(DATA_DIR + path):
            shutil.rmtree(DATA_DIR + path)
        else:
            os.remove(DATA_DIR + path)


def clear_spectral_data(DATA_DIR):
    """Clears spectral data from the data directory."""
    print('Clearing spectral data')
    contents = os.listdir(DATA_DIR)
    for path in contents:
        if path == 'spectra':
            shutil.rmtree(DATA_DIR + path)


def get_library(DATA_DIR, file=False):
    if file:
        print('Loading USGS Spectral Library from file')
        filename = DATA_DIR + 'spectra.zip'
        shutil.copyfile(file, filename)
    else:
        print('Downloading the USGS Spectral Library Verson 7')
        url = 'http://www.sciencebase.gov/catalog/file/get/586e8c88e4b0f5ce109fccae/?f=__disk__a7%2F4f%2F91%2Fa74f913e0b7d1b8123ad059e52506a02b75a2832'
        filename = 'cycsat/data/spectra.zip'
        filename = wget.download(url, filename)

    print('')
    print('Unpacking...')
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('cycsat/data/')
    zip_ref.close()


def detect_scale(file):
    if 'ASD' in file:
        return pd.read_table(
            DATASET + 'splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt',
            skiprows=1, header=None)
    elif 'AVIRIS' in file:
        return pd.read_table(
            DATASET + 'splib07a_Wavelengths_AVIRIS_1996_0.37-2.5_microns.txt',
            skiprows=1, header=None)
    elif 'BECK' in file:
        return pd.read_table(
            DATASET + 'splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt',
            skiprows=1, header=None)
    elif 'NIC':
        return pd.read_table(
            DATASET + 'splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt',
            skiprows=1, header=None)


def learn_lib(DATA_DIR, DATASET):
    # the chapters of the spectra library
    chapters = ['ChapterA_ArtificialMaterials',
                'ChapterC_Coatings',
                'ChapterL_Liquids',
                'ChapterM_Minerals',
                'ChapterO_OrganicCompounds',
                'ChapterS_SoilsAndMixtures',
                'ChapterV_Vegetation']

    print('Reading data files')
    sensors = {}
    for chap in chapters:
        fs = os.listdir(DATASET + chap)
        for f in fs:
            scale = detect_scale(f)
            data = pd.read_table(DATASET + chap + '//' +
                                 f, skiprows=1, header=None)
            sensors[f.replace('splib07a_', '')] = scale.assign(data=data)

    os.mkdir('cycsat/data/spectra')
    for i, material in enumerate(sensors, start=1):
        df = sensors[material]

        X = df[0].values.reshape(-1, 1)
        y = df['data']

        model = KNeighborsRegressor()
        model.fit(X, y)
        pickle.dump(model, open('cycsat/data/spectra/' + material, 'wb'))

        sys.stdout.write("Fitting models: %d%%   \r" %
                         (round((i / len(sensors)) * 100, 2)))
        sys.stdout.flush()

    print('Fitting models: 100%')
    print('Cleaning up')

    shutil.rmtree(DATASET)
    os.remove(DATA_DIR + 'spectra.zip')


def compile_spectra(DATA_DIR, DATASET, file=False):
    clear_spectral_data(DATA_DIR)
    get_library(DATA_DIR, file)
    learn_lib(DATA_DIR, DATASET)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=False,
                        help="Loads the USGS library from file (optional).")
    args = parser.parse_args()

    compile_spectra(DATA_DIR, DATASET, args.file)
