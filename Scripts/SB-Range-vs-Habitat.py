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
    
    It also requires a text file of 12-digit range HUCs (HUC12s.txt)
    that contains data on each HUC's areal extent for calculating total
    and proportional area of range extent.
    
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
        BytesIO
        Seaborn
        skikit-learn (or sklearn)
        requests
        datetime


@author: mjrubino
25 February 2019

"""

##############################################################################
def GetIndex(lst, k, v):
    """
    Gets the numeric index position of an item in a list
    of dictionaries based on a value string for a given key
    
    """
    for i, dic in enumerate(lst):
        if dic[k] == v:
            return i
    return -1

##############################################################################
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
    flst = item_json['files']
    zname = '{0}_CONUS_Range_2001v1.zip'.format(gap_id)
    # Use the GetIndex function to find the zip file's index value in the 
    # JSON item's files list dictionaries of name keys
    zip_index = GetIndex(flst,'name',zname)
    # Here's a way to do this without using the GetIndex function created above
    #zip_index=next((index for (index, d) in enumerate(flst) if d["name"] == zname), None)
    # Get the URL to the zip file containing the HUC CSV
    rngzipURL = item_json['files'][zip_index]['url']
    r = requests.get(rngzipURL)
    z = zipfile.ZipFile(BytesIO(r.content))

    # Get ONLY the HUC CSV file and extract it to the designated directory
    rngCSV = [y for y in sorted(z.namelist()) for end in ['csv'] if y.endswith(end)]
    csvFile = z.extract(rngCSV[0], toDir)
    z.close()
    
    # Return the extracted range CSV
    return csvFile

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
    flst = item_json['files']
    zname = '{0}_CONUS_HabMap_2001v1.zip'.format(gap_id)
    # Use the GetIndex function to find the zip file's index value in the 
    # JSON item's files list dictionaries of name keys
    zip_index = GetIndex(flst,'name',zname)
    # Here's a way to do this without using the GetIndex function created above
    #zip_index=next((index for (index, d) in enumerate(flst) if d["name"] == zname), None)
    # Get the URL to the zip file containing the VAT dbf
    habzipURL = item_json['files'][zip_index]['url']
    r = requests.get(habzipURL)
    z = zipfile.ZipFile(BytesIO(r.content))

    # Get ONLY the VAT dbf file and extract it to the designated directory
    habDBF = [y for y in sorted(z.namelist()) for end in ['dbf'] if y.endswith(end)]
    dbfFile = z.extract(habDBF[0], toDir)
    z.close()

    # Return path to VAT dbf file
    return dbfFile

##############################################################################
##############################################################################
def GetHabitatArea(spcode, tDir):
    
    import glob
    from simpledbf import Dbf5

    ''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Get HABITAT data based on species' model outputs from SB file downloads
    '''
    print("\n\n++++++++++++++ Running calculations based on HABITAT +++++++++++++++++\n")
    
    # Get the name of the raster VAT dbf file for data on the number
    # of habitat cells in the species habitat map
    habDBF = glob.glob(tDir + '{0}*tif.vat.dbf'.format(spcode))[0]
    # Make it a dataframe using simpledbf
    dbf = Dbf5(habDBF)
    dfDBF = dbf.to_dataframe()
    # Make sure all the column names are UPPERCASE - some vat dbf's have a mixture of cases
    # Calculations using column names that aren't exact will throw a KeyError
    dfDBF.columns = map(str.upper, dfDBF.columns)
    
    print("---> Calculating total habitat area and proportion ....")
    # Make and empty data list for the species habitat count data
    cntdata = []
    # Calculate the km2 area for the species habitat count data
    #  the proportion of CONUS and add the species code as index
    cntsum = dfDBF['COUNT'].sum()
    cntdata.append({'SpeciesCode':spcode,
                    'AreaHab_km2':cntsum * 0.0009,
                    'PropHab_CONUS':cntsum/cntLC
                   })
    # Append the data to a dataframe that will be joined with HUC 12 data
    dfHabCounts = pd.DataFrame(data=cntdata)
    dfHabCounts = dfHabCounts[['SpeciesCode','AreaHab_km2','PropHab_CONUS']]
    dfHabArea = dfHabCounts.set_index(keys=['SpeciesCode'])
   
    return dfHabArea

##############################################################################
##############################################################################
def GetRangeArea(spcode, tDir):
    
    import glob
    
    ''' ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Get HUC range data from the Species Database from SB file downloads
    '''
    print("\n++++++++++++++ Running calculations based on RANGES +++++++++++++++++\n")
    
    # This gets the name of the extracted range CSV file
    rangeCSV = glob.glob(tDir + '{0}*.csv'.format(spcode))[0]
    # make it a dataframe
    dfRangeCSV = pd.read_csv(rangeCSV,dtype={'strHUC12RNG':object},sep=',')
    # Select only known, possibly, or potentially present;
    #             year-round, winter, or summer seasons
    select={'intGapPres':[1,2,3], 'intGapSeas':[1,3,4]}
    dfS1 = dfRangeCSV[dfRangeCSV[list(select)].isin(select).all(axis=1)]
    
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
    dfSppHUCs = pd.merge(left=dfS1, right=dfHUCsTable, how='inner',
     left_on='strHUC12RNG', right_on='HUC12RNG')
    
    # Get a row and column count from the above dataframe
    # The number of rows is the total number of HUCs in the species' range
    (r,c) = dfSppHUCs.shape
    
    # Sum the total area of each HUC (this is in km2)
    sSum = dfSppHUCs.groupby(['strUC'])['AreaRange_km2'].sum()
    dfRangeArea = pd.DataFrame(data=sSum)
    dfRangeArea.index.name = 'SpeciesCode'
    
    # Add a field with the total number of HUCs in the species' range
    print("---> Calculating number of HUCs in species' range ....")
    dfRangeArea['nHUCs'] = r
    # Add a field to calculate the proportion of CONUS for the species' range
    print("---> Calculating proportion of species' range in CONUS ....")
    dfRangeArea['Prop_CONUS'] = dfRangeArea['AreaRange_km2']/CONUSArea
    # Reorder columns
    dfRangeArea = dfRangeArea[['AreaRange_km2','nHUCs','Prop_CONUS']]
    
    return dfRangeArea

##############################################################################


import os, sys, shutil, sciencebasepy
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO


analysisDir = 'C:/Data/USGS Analyses/'
workDir = analysisDir + 'Range-vs-Habitat/'
tempDir = workDir + 'downloadtemp/'
# ****** Static Range HUCs Table **********
HUCfile = workDir + 'HUC12s.txt'

starttime = datetime.now()
timestamp = starttime.strftime('%Y-%m-%d')


# Make temporary directory for downloads
#  remove it if it already exists
if os.path.exists(tempDir):
    shutil.rmtree(tempDir)
    os.mkdir(tempDir)
else:
    os.mkdir(tempDir)


'''
    Function to write an error log if a species' ScienceBase
    range or habitat file connection cannot be made
'''
log = workDir + 'Species-Data-Access-Error-Log.txt'
def Log(content):
    with open(log, 'a') as logDoc:
        logDoc.write(content + '\n')

# STATIC VARIABLES
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
sppCodeList = ['mAHNSx']

## Here is a way to limit rows based on partial text strings in a column
#   in this example, amphibians where the first letter in the 4 part code is B
#sppCodeList = dfSppList[dfSppList['GAP_code'].str.contains("aB")==True]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

    Run the functions that pull GAP species range and habitat
    info and download files to a temporary location
    
    Loop over a species code list

'''

for sppCode in sppCodeList:
    try:
        
        print('\n')    
        print('*'*85)
        print('RUNNING THE FOLLOWING SPECIES CODE  >>>',sppCode,'<<<' )
        
        # Get the scientific and common names from the SB csv file
        SN = dfSppCSV.loc[dfSppCSV['GAP_code']==sppCode,'scientific_name'].item()
        CN = dfSppCSV.loc[dfSppCSV['GAP_code']==sppCode,'common_name'].item()
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
        # Run the GetHabitatArea function
        dfHab = GetHabitatArea(sppCode, tempDir)
        
        # Run the GetRangeArea function
        dfRange = GetRangeArea(sppCode, tempDir)
        
        # Merge the habitat dataframe returned from the GetHabitatArea function
        # with the range dataframe returned from the GetRangeArea function
        print("\nMerging Range-based and Habitat-based Dataframes ....\n")
        dfMerge = pd.merge(left=dfRange, right=dfHab,
                       how='inner', left_index=True, right_index=True)
        # Add scientific and common names   
        dfMerge['ScientificName'] = SN
        dfMerge['CommonName'] = CN
        # Add Log10 transformed area columns for range and habitat
        dfMerge['LogAreaRange'] = np.log10(dfMerge['AreaRange_km2'])
        dfMerge['LogAreaHabitat'] = np.log10(dfMerge['AreaHab_km2'])
        
        # Append to the master dataframe
        dfMaster = dfMaster.append(dfMerge, ignore_index=False)
        
        # Delete the global scientific and common name variables to avoid overlap
        #del SN,CN
        #del habDBF,dbf,dfDBF,rangeCSV,dfRangeCSV,dfHUCsTable,dfSppHUCs
        #del r,c,sSum,dfSppSum, dfMerge
        
        # Delete the temporary download directory
        if os.path.exists(tempDir):
            shutil.rmtree(tempDir)
            os.mkdir(tempDir)
    
    except:
        print('\n!!!! Had Problems With Connections to ScienceBase. Moving on to Next Species ...!!!!')
        Log(sppCode)
    
    
    

# Export to CSV
print('*'*85)
print('\nExporting to CSV: SpeciesRangevsHabitat.csv\n')
dfMaster.to_csv(workDir + "SpeciesRangevsHabitat.csv")


endtime = datetime.now()
delta = endtime - starttime
print("+"*35)
print("Processing time: " + str(delta))
print("+"*35)

'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
Now start manipulating the dataframe to plot log areas for range and habitat

'''

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


## -- Pull out only the species code, log area range, and log area habitat
# reset the index so the species code is no longer the index
dfPlot = dfMaster[['LogAreaRange','LogAreaHabitat']].reset_index()
#dfPlot = dfMaster[['AreaRange_km2','AreaHab_km2']].reset_index()

# create a new Taxon column based on the first letter in the species code
dfPlot['Taxon'] = np.where(dfPlot['SpeciesCode'].str[:1]=='a', 'Amphibians',
		  np.where(dfPlot['SpeciesCode'].str[:1]=='b', 'Birds',
		  np.where(dfPlot['SpeciesCode'].str[:1]=='m', 'Mammals', 'Reptiles')))


a = sns.lmplot(x="LogAreaRange", y="LogAreaHabitat", 
               hue="Taxon", data=dfPlot, fit_reg=False, legend=False,
               markers=['o','v','s','D'], size=10,
               scatter_kws={'s': 55})
#a = sns.lmplot(x="AreaRange_km2", y="AreaHab_km2", 
#               hue="Taxon", data=dfPlot, fit_reg=False, legend=False,
#               markers=['o','v','s','D'], size=10,
#               scatter_kws={'s': 55})

pax=sns.regplot(x="LogAreaRange", y="LogAreaHabitat", 
            data=dfPlot, scatter=False, ax=a.axes[0, 0],
            line_kws={"color":"black","alpha":0.5,"lw":1})
#pax=sns.regplot(x="AreaRange_km2", y="AreaHab_km2", 
#            data=dfPlot, scatter=False, ax=a.axes[0, 0],
#            line_kws={"color":"black","alpha":0.5,"lw":1})

pax.set_xlabel('log10 Range Area', fontsize=10)
pax.set_ylabel('log10 Habitat Area', fontsize=10)

#pax.set_xlabel('Range Area km2', fontsize=10)
#pax.set_ylabel('Habitat Area km2', fontsize=10)
#pax.set_xscale("log")
#pax.set_yscale("log")

# Move the legend to an empty part of the plot
lgd = plt.legend(loc='lower right', title='Taxon', prop={'size':12})
lgd.get_title().set_fontsize(14)


### Make individual taxa subplots on a 2x2 figure ###

fig, axs = plt.subplots(2, 2, figsize=(11, 11))
axs = axs.flatten()
fig.text(0.5, 0.04, 'log10 Range Area', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'log10 Habitat Area', va='center', rotation='vertical',fontsize=14)


# Hard code a taxa list
tlst = ['Amphibians', 'Birds', 'Mammals', 'Reptiles']
# Make a list of markers
mlst = ['o','v','s','D']
# Add an iterator variable
i=0
for ax, t, m in zip(axs, tlst, mlst):

    # Pull out the taxa specific data from dfPlot
    dfT = dfPlot[dfPlot['Taxon']==t]
    
    # Set an axes title
    ax.set_title(t)
    
    # Use the default color palette from Seaborn as variables
    c = sns.color_palette()[i]
    
    # Plot a scatter plot and add a regression trend line
    # Use the Seaborn colors in the plot with all taxa for individual taxa
    ax.scatter(x=dfT['LogAreaRange'],y=dfT['LogAreaHabitat'],marker=m,c=c)
    model = LinearRegression(fit_intercept=True)
    model.fit(dfT['LogAreaRange'][:, np.newaxis], y = dfT['LogAreaHabitat'])
    xfit = dfT['LogAreaRange']
    yfit = model.predict(xfit[:, np.newaxis])
    ax.plot(xfit, yfit,linewidth=1.5,color='black')
    
    i+=1

del fig,axs,tlst,mlst,i,ax,t,m,dfT,c,xfit,yfit


# Run an OLS regression of habitat and range and get an r-squared for the relationship
lm = sm.OLS.from_formula(formula='LogAreaHabitat ~ LogAreaRange', data=dfPlot)
result = lm.fit()
print(result.summary())
r2=result.rsquared
print('\n The r-squared for log habitat area given log range area =', r2)


