import uproot 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import os.path

'''Takes in a root file and extracts the appropriate data, then uses uproot to view the 2D Histogram image.'''

Sample = 'TTbar'
BASE_PATH = "/storage/ec6821/L1TJets/MsciProjects/2020/CMSSW_10_6_1_patch2/src/L1Trigger/L1CaloTrigger/test/Histograms_{sample}/".format(sample=Sample) 

for subdir, dirs, files in os.walk(BASE_PATH):
    i = 0
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".root"):
            f = uproot.open(filepath)
            a = f["caloGrid"].values
            # plt.imshow(a)
            # plt.show()
            outPathBase = '{sample}_numpy'.format(sample=Sample.upper())
            if not os.path.isdir(outPathBase):
                os.mkdir(outPathBase)
            outFile = outPathBase + '/' + filename.replace('.root','')
            np.save(outFile,a)
            # break