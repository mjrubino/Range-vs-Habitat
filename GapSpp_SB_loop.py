# -*- coding: utf-8 -*-
"""
GapSpp_SB_loop.py

enviroment: Python3

Retrieves GAP species list from SB Habitat Map item and loops through each.
 
Created on Wed Oct 24 17:07:53 2018

@author: sgwillia
=======================================================================
"""
# Import sbconfig variables containing SB user/password
import sys
sys.path.append('C:/Code')
import sbconfig
# Import other packages
import pandas as pd
from sciencebasepy import SbSession
from io import StringIO

# =======================================================================
# LOCAL VARIABLES
home = "N:/Git_Repositories/fetchSppInfo/"
outDir = home + 'itis'

# =======================================================================
# LOCAL FUNCTIONS 
## -------------------Connect to ScienceBase-------------------------
def ConnectToSB(username=sbconfig.sbUserName, password=sbconfig.sbWord):
    '''
    (string) -> connection to ScienceBase
    
    Creats a connection to ScienceBase. You will have to enter your password.
    
    Arguments:
    username -- your ScienceBase user name.
    password -- your ScienceBase password.
    '''

    sb = SbSession()
    sb.login(username, password)
    return sb

# =======================================================================
# GET SPP TABLE FROM SB HABMAP ITEM
sb = ConnectToSB()
habItem = sb.get_item("527d0a83e4b0850ea0518326")
for file in habItem["files"]:
    if file["name"] == "ScienceBaseHabMapCSV_20180713.csv":
        tbSpp = pd.read_csv(StringIO(sb.get(file["url"])))
   
# =======================================================================
# SET UP LOOP ON SPECIES
# Iterate through table
for row in tbSpp.itertuples(index=True, name='Pandas'):    
    # set variables
    strUC = getattr(row,"GAP_code")
    strSciName = getattr(row,"scientific_name")
    strComName = getattr(row,"common_name")
    intITIScode = getattr(row,"TSN_code")
    strSbHabID = getattr(row,"ScienceBase_url")[-24:]
        
    print(strUC)
    print('  ' + strSciName)
    print('  ' + strComName)
    print('  ' + str(intITIScode))
    print('  ' + strSbHabID)
