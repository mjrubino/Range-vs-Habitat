"""

                        SB-Range-vs-Habitat.py
        
    Use this to create a CSV file of species' (including subspecies)
    HUC12 range sizes - in km2 and number of HUCs - and the proportion
    that range relative to the totatl CONUS area. The identified range
    includes both breeding and non-breeding and summer and winter
    seasons. It DOES NOT INCLUDE any migratory portions of a species'
    range nor does it include any historic and/or extirpated portions
    that may have been delineated.
    Additionally, the script will calculate "range" (extent really) based
    on HABITAT as oppossed to HUC 12 range. This uses the habitat maps
    with NoData and 1,2, and/or 3s and calculates as sum as well as
    proportion relative to the CONUS land cover (excluding 0s). Note
    that total CONUS land cover area could exclude water. The script
    sets variables for both water included and water excluded cell
    counts - cntLC and cntLCnoW respectively.
    
    This uses the sciencebasepy package and local functions to download
    and unzip GAP range and habitat species data from ScienceBase.
    
    OUTPUT CSV FILE NAME: SpeciesRangevsHabitat.csv
    
    The final CSV file will contain the following fields:
    SpeciesCode
    ScientificName
    CommonName
    AreaRange_km2
    nHUCS
    Prop_CONUS
    AreaHab_km2
    PropHab_CONUS
    LogAreaRange
    LogAreaHabitat
    

    Package dependancies:
        sciencebasepy
        glob
        zipfile
        pandas
        simpledbf
        numpy
        datetime (for calculating processing time)
        StringIO


@author: mjrubino
25 February 2019

"""

##############################################################################
def download_GAP_range_CONUS2001v1(gap_id, toDir):
    """
    Downloads GAP Range CONUS 2001 v1 file and returns path to the unzipped
    file.  NOTE: doesn't include extension in returned path so that you can
    specify if you want csv or shp or xml when you use the path.
    """
    import sciencebasepy
    import zipfile
    import requests
    from io import BytesIO

    # Connect
    sb = sciencebasepy.SbSession()

    # Search for gap range item in ScienceBase
    gap_id = gap_id[0] + gap_id[1:5].upper() + gap_id[5]
    item_search = '{0}_CONUS_2001v1 Range Map'.format(gap_id)
    items = sb.find_items_by_any_text(item_search)

    # Get a public item.  No need to log in.
    rngID =  items['items'][0]['id']
    item_json = sb.get_item(rngID)
    rngzipURL = item_json['files'][0]['url']
    r = requests.get(rngzipURL)
    z = zipfile.ZipFile(BytesIO(r.content))
    #get_files = sb.get_item_files(item_json, toDir)

    # Unzip
    #rng_zip = toDir + item_json['files'][0]['name']
    #zip_ref = zipfile.ZipFile(rng_zip, 'r')
    # Get ONLY the VAT dbf file and extract it to the designated directory
    rngCSV = [y for y in sorted(z.namelist()) for end in ['csv'] if y.endswith(end)]
    csvFile = z.extract(rngCSV[0], toDir)
    #zip_ref.extractall(toDir)
    z.close()
    
    # Return the extracted range CSV
    return csvFile

    # Return path to range file without extension
    #return rng_zip.replace('.zip', '')
##############################################################################
##############################################################################
def download_GAP_habmap_CONUS2001v1(gap_id, toDir):
    """
    Downloads GAP Habitat Map CONUS 2001 v1 file and returns path to the unzipped
    file.  NOTE: doesn't include extension in returned path so that you can
    specify if you want csv or shp or xml when you use the path.
    """
    import sciencebasepy
    import zipfile
    import requests
    from io import BytesIO

    # Connect
    sb = sciencebasepy.SbSession()

    # Search for gap range item in ScienceBase
    gap_id = gap_id[0] + gap_id[1:5].upper() + gap_id[5]
    item_search = '{0}_CONUS_2001v1 Habitat Map'.format(gap_id)
    items = sb.find_items_by_any_text(item_search)

    # Get a public item.  No need to log in.
    habID =  items['items'][0]['id']
    item_json = sb.get_item(habID)
    habzipURL = item_json['files'][2]['url']
    r = requests.get(habzipURL)
    z = zipfile.ZipFile(BytesIO(r.content))
    #get_files = sb.get_item_files(item_json, toDir)
    
    # Set global Scientific and Common name variables from item JSON
    global CN 
    CN = item_json['identifiers'][1]['key']
    global SN
    SN = item_json['identifiers'][2]['key']

    # Unzip
    #hab_zip = toDir + item_json['files'][2]['name']
    #zip_ref = zipfile.ZipFile(hab_zip, 'r')
    # Get ONLY the VAT dbf file and extract it to the designated directory
    habDBF = [y for y in sorted(z.namelist()) for end in ['dbf'] if y.endswith(end)]
    dbfFile = z.extract(habDBF[0], toDir)
    z.close()
    #zip_ref.extractall(toDir)
    #zip_ref.close()

    # Return path to VAT dbf file
    return dbfFile
    #return hab_zip.replace('.zip', '')

##############################################################################

import glob, os, sys, shutil, sciencebasepy
import pandas as pd
import numpy as np
from simpledbf import Dbf5
from datetime import datetime
from io import StringIO

analysisDir = 'C:/Data/USGS Analyses/'
richdataDir = analysisDir + 'Richness/data/'
workDir = 'C:/Data/temp/'
tempDir = workDir + 'downloadtemp/'
# ****** Static Range HUCs Table **********
HUCfile = richdataDir + 'HUC12s.txt'

starttime = datetime.now()
timestamp = starttime.strftime('%Y-%m-%d')


# Make temporary directory for downloads
#  remove it if it already exists
if os.path.exists(tempDir):
    shutil.rmtree(tempDir)
    os.mkdir(tempDir)
else:
    os.mkdir(tempDir)


CONUSArea = 8103534.7   # 12-Digit HUC CONUS total area in km2
nHUCs = 82717.0         # Number of 12-digit HUCS in CONUS
cntLC = 9000763993.0    # Cell count of CONUS landcover excluding 0s
cntLCnoW = 8501572144.0 # Cell count of CONUS landcover excluding 0s and water

# Make an empty master dataframe
dfMaster = pd.DataFrame()

'''
    Connect to ScienceBase to pull down a species list
    This uses the ScienceBase item for species habitat maps
    and searches for a CSV file with species info in it.
    The habitat map item has a unique id (527d0a83e4b0850ea0518326)
    and the CSV file is named ScienceBaseHabMapCSV_20180713.csv. If
    either of these change, the code will need to be re-written.

'''
sb = sciencebasepy.SbSession()
habmapItem = sb.get_item("527d0a83e4b0850ea0518326")
for file in habmapItem["files"]:
    if file["name"] == "ScienceBaseHabMapCSV_20180713.csv":
        dfSppCSV = pd.read_csv(StringIO(sb.get(file["url"])))


# Check to make sure the CSV file was returned
if dfSppCSV is not None:
    print('-'*55)
    print('== Found ScienceBase Habitat Maps Item CSV File ==')
    print('-'*55)
else:
    print('!!! Could Not Find ScienceBase CSV File. Exiting !!!')
    sys.exit()

# Pull out only scientific and common names and species codes
dfSppList = dfSppCSV[['GAP_code','scientific_name','common_name']]
# Pull out species codes for looping over
# NOTE: this is a series not a dataframe
#sppCodeList = dfSppList['GAP_code']

## Here is a way to limit rows based on partial text strings in a column
#   in this example, amphibians where the first letter in the 4 part code is B
#sppCodeList = dfSppList[dfSppList['GAP_code'].str.contains("aB")==True]
sppCodeList = ['aACSSx','aBESAx','aBLASx']

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

    Run the functions that pull GAP species range and habitat
    info and download files to a temporary location
    
    Loop over a species code list

'''

for sppCode in sppCodeList:
    print('\n')    
    print('*'*85)
    print('RUNNING THE FOLLOWING SPECIES CODE  >>>',sppCode,'<<<' )
    # Run the download GAP range function
    print('\nRunning function to download GAP RANGE from ScienceBase\n' )
    download_GAP_range_CONUS2001v1(sppCode, tempDir)
    
    
    # Run the download GAP habitat map function
    print('\nRunning function to download GAP HABITAT from ScienceBase\n' )
    download_GAP_habmap_CONUS2001v1(sppCode, tempDir)
    
    '''
    
    Start manipulating the files in the download directory after
    unzipping to get out the necessary information on range and habitat
    
    '''
    ''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Get HABITAT data based on species' model outputs from SB file downloads
    '''
    print("\n\n++++++++++++++ Running calculations based on HABITAT +++++++++++++++++\n")
    
    # Get the name of the raster VAT dbf file for data on the number
    # of habitat cells in the species habitat map
    habDBF = glob.glob(tempDir + '{0}*tif.vat.dbf'.format(sppCode))[0]
    # Make it a dataframe using simpledbf
    dbf = Dbf5(habDBF)
    dfDBF = dbf.to_dataframe()
    # Make sure all the column names are UPPERCASE - some vat dbf's have a mixture of cases
    # Calculations using column names that aren't exact will throw a KeyError
    dfDBF.columns = map(str.upper, dfDBF.columns)
    
    print("---> Calculating total habitat area and proportion ....")
    # Calculate the km2 area for the species habitat count data
    #  the proportion of CONUS and add the species code as index
    dfDBF['AreaHab_km2'] = dfDBF['COUNT'].sum() * 0.0009
    dfDBF['PropHab_CONUS'] = dfDBF['COUNT'].sum()/cntLC
    # Drop VALUE and COUNT fields
    dfDBF = dfDBF.drop(['VALUE','COUNT'], axis=1)
    dfDBF['SpeciesCode'] = sppCode
    dfHab = dfDBF.set_index(keys=['SpeciesCode'])
    
    ''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Get HUC range data from the Species Database from SB file downloads
    '''
    print("\n++++++++++++++ Running calculations based on RANGES +++++++++++++++++\n")
    
    # This gets the name of the extracted range CSV file
    rangeCSV = glob.glob(tempDir + '{0}*.csv'.format(sppCode))[0]
    # make it a dataframe
    dfRangeCSV = pd.read_csv(rangeCSV,dtype={'strHUC12RNG':object},sep=',')
    
    # Pull in data from the HUC shapefile table (currently a static
    #  text file located at D:/USGS Analyses/Richness/data/HUCs12.txt)
    dfHUCsTable = pd.read_csv(HUCfile, dtype={'HUC12RNG':object},thousands=',',decimal='.')
    # Add an area field calculated in km2
    print("---> Calculating range area in km2 ....")
    dfHUCsTable['AreaRange_km2'] = dfHUCsTable['Shape_Area']/1000000
    # Drop some unnecessary fields
    dfHUCsTable = dfHUCsTable.drop(['FID','HUC_10','HUC_12','HUC_8', 'OBJECTID',
    'STATES','Shape_Area','Shape_Leng'], axis=1)
    
    # Now merge species range HUCs dataframe with HUC shapefile dataframe
    dfSppHUCs = pd.merge(left=dfRangeCSV, right=dfHUCsTable, how='inner',
     left_on='strHUC12RNG', right_on='HUC12RNG')
    
    # Get a row and column count from the above dataframe
    # The number of rows is the total number of HUCs in the species' range
    (r,c) = dfSppHUCs.shape
    
    # Sum the total area of each HUC (this is in km2)
    sSum = dfSppHUCs.groupby(['strUC'])['AreaRange_km2'].sum()
    dfSppSum = pd.DataFrame(data=sSum)
    dfSppSum.index.name = 'SpeciesCode'
    
    # Add a scientific and common names   
    dfSppSum['ScientificName'] = SN
    dfSppSum['CommonName'] = CN
    # Add a field with the total number of HUCs in the species' range
    print("---> Calculating number of HUCs in species' range ....")
    dfSppSum['nHUCs'] = r
    # Add a field to calculate the proportion of CONUS for the species' range
    print("---> Calculating proportion of species' range in CONUS ....")
    dfSppSum['Prop_CONUS'] = dfSppSum['AreaRange_km2']/CONUSArea
    # Reorder columns
    dfSppSum = dfSppSum[['ScientificName','CommonName',
     'AreaRange_km2','nHUCs','Prop_CONUS']]
    
    # Finally, merge with the master dataframe
    print("\nMerging Range-based and Habitat-based Dataframes ....\n")
    dfMerge = pd.merge(left=dfSppSum, right=dfHab,
                   how='inner', left_index=True, right_index=True)
    # Add Log10 transformed area columns for range and habitat
    dfMerge['LogAreaRange'] = np.log10(dfMerge['AreaRange_km2'])
    dfMerge['LogAreaHabitat'] = np.log10(dfMerge['AreaHab_km2'])
    
    # Append to the master dataframe
    dfMaster = dfMaster.append(dfMerge, ignore_index=False)
    
    # Delete the global scientific and common name variables to avoid overlap
    del SN,CN
    del habDBF,dbf,dfDBF,rangeCSV,dfRangeCSV,dfHUCsTable,dfSppHUCs
    del r,c,sSum,dfSppSum, dfMerge
    
    # Delete the temporary download directory
    if os.path.exists(tempDir):
        shutil.rmtree(tempDir)
        os.mkdir(tempDir)
    
# Export to CSV
print('*'*85)
print('\nExporting to CSV: SpeciesRangevsHabitat.csv\n')
dfMaster.to_csv(workDir + "SpeciesRangevsHabitat.csv")


endtime = datetime.now()
delta = endtime - starttime
print("+"*35)
print("Processing time: " + str(delta))
print("+"*35)





