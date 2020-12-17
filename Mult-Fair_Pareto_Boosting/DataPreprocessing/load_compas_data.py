from __future__ import division
#import urllib2
import os,sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import preprocessing
from random import seed, shuffle

# import utils as ut
import collections

SEED = 1234
seed(SEED)
np.random.seed(SEED)

"""
    The adult dataset can be obtained from: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv
    The code will look for the data file in the present directory, if it is not found, it will download them from GitHub.
"""

def check_data_file(fname):
    files = os.listdir(".") # get the current directory listing
    print ("Looking for file '%s' in the current directory..." % fname)
    '''
    if fname not in files:
        print ("'%s' not found! Downloading from GitHub..." % fname)
        addr = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        response = urllib2.urlopen(addr)
        data = response.read()
        fileOut = open(fname, "w")
        fileOut.write(data)
        fileOut.close()
        print "'%s' download and saved locally.." % fname
    else:
        print "File found in current directory.."
    '''

def load_compas():

	FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count", "c_charge_degree"] #features to be used for classification
	CONT_VARIABLES = ["priors_count"] # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
	CLASS_FEATURE = "two_year_recid" # the decision variable
	SENSITIVE_ATTRS = ['race','sex']# ['age_cat','race','sex']#
	p_group = [0 for v in SENSITIVE_ATTRS]#[0,0,0]

	# COMPAS_INPUT_FILE = "compas-scores-two-years.csv"
	COMPAS_INPUT_FILE = "DataPreprocessing/compas-scores-two-years.csv"
	# check_data_file(COMPAS_INPUT_FILE)

	# load the data and get some stats
	df = pd.read_csv(COMPAS_INPUT_FILE)
	df = df.dropna(subset=["days_b_screening_arrest"]) # dropping missing vals
	
	# convert to np array
	data = df.to_dict('list')
	for k in data.keys():
		data[k] = np.array(data[k])


	""" Filtering the data """

	# These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
	# If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense. 
	idx = np.logical_and(data["days_b_screening_arrest"]<=30, data["days_b_screening_arrest"]>=-30)


	# We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
	idx = np.logical_and(idx, data["is_recid"] != -1)

	# In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
	idx = np.logical_and(idx, data["c_charge_degree"] != "O") # F: felony, M: misconduct

	# We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
	idx = np.logical_and(idx, data["score_text"] != "NA")

	# we will only consider blacks and whites for this analysis
	idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

	# select the examples that satisfy this criteria
	for k in data.keys():
		data[k] = data[k][idx]



	#print (collections.Counter(data[sa]))

	test = pd.DataFrame.from_dict(data)
	# print test
	""" Feature normalization and one hot encoding """

	# convert class label 0 to -1
	y = data[CLASS_FEATURE]
	y[y==0] = -1



	#print "\n"


	X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
	x_control = defaultdict(list)

	feature_names = []
	index = -1
	saIndex = 0
	for attr in FEATURES_CLASSIFICATION:
		index +=1

		vals = data[attr]
		#if attr=='age_cat':
		#	vals=[1 if v=='age_cat_25 - 45' else 0 for v in vals]
		if attr in CONT_VARIABLES:
			vals = [float(v) for v in vals]
			vals = preprocessing.scale(vals) # 0 mean and 1 variance  
			vals = np.reshape(vals, (len(y), -1)) # convert from 1-d arr to a 2-d arr with one col

		else: # for binary categorical variables, the label binarizer uses just one var instead of two
			lb = preprocessing.LabelBinarizer()
			lb.fit(vals)
			vals = lb.transform(vals)

		# add to sensitive features dict
		if attr in SENSITIVE_ATTRS:
			x_control[attr] = vals


		# add to learnable features
		X = np.hstack((X, vals))

		if attr in CONT_VARIABLES: # continuous feature, just append the name
			feature_names.append(attr)
		else: # categorical features
			if vals.shape[1] == 1: # binary features that passed through lib binarizer
				feature_names.append(attr)
			else:
				for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
					feature_names.append(attr + "_" + str(k))


	# convert the sensitive feature to 1-d array
	x_control = dict(x_control)
	for k in x_control.keys():
		assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
		x_control[k] = np.array(x_control[k]).flatten()

	# sys.exit(1)

	"""permute the date randomly"""
	perm = list(range(0,X.shape[0]))
	shuffle(perm)
	X = X[perm]
	y = y[perm]
	for k in x_control.keys():
		x_control[k] = x_control[k][perm]
	feature_names.append('target')
	print( "Features we will be using for classification are:", feature_names, "\n")

	# pd.DataFrame(np.c_[X, y]).to_csv("test_compas_X.csv", header=feature_names)
	# print np.sum(X[:,feature_names.index(SENSITIVE_ATTRS[0])])

	return X, y, [feature_names.index(SENSITIVE_ATTRS[i]) for i in range(len(SENSITIVE_ATTRS))], p_group, x_control,feature_names




'''
from load_compas_data import load_compas
import numpy as np
X=[]
for i in range(len(X1)):
    v=0
    m=X1[i,2]
    if m==1:
        v=0
    elif m==0:
        v=1
    r=[v]+list(X1[i][3:])
    X.append(np.array(r))

X=np.array(X)



width = 0.35

labels = ['race','gender','age']
x = np.arange(len(labels))
pp=[0.55,0.54,0.55]
npp=[0.53,0.55,0.53]
pn=[0.74,0.73,0.72]
npn=[0.72,0.73,0.77]

adpp=[0.64,0.48,0.64]
adnpp=[0.58,0.65,0.53]
adpn=[0.67,0.78,0.64]
adnpn=[0.66,0.63,0.75]

abpp=[0.65,0.59,0.57]
abnpp=[0.38,0.67,0.64]
abpn=[0.67,0.76,0.76]
abnpn=[0.88,0.58,0.7]

ppa=[0.53,0.52,0.52]
nppa=[0.5,0.52,0.52]
pna=[0.75,0.75,0.74]
npna=[0.75,0.75,0.78]

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=[18.8,9.6])
rects1 = ax1.bar(x - width, pp, width/2, label='protpos')
rects2 = ax1.bar(x - width/2, npp, width/2, label='nonprot-pos') 
rects3 = ax1.bar(x + width/8, pn, width/2, label='protneg')
rects4 = ax1.bar(x + width/1.5, npn, width/2, label='nonprot-neg') 
plt.ylim(top=1)
ax1.set_ylabel('acc')
ax1.set_title('TPR n TNR MultiFair max compass')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(prop={"size":7})

rects1 = ax2.bar(x - width, adpp, width/2, label='protpos')
rects2 = ax2.bar(x - width/2, adnpp, width/2, label='nonprot-pos') 
rects3 = ax2.bar(x + width/8, adpn, width/2, label='protneg')
rects4 = ax2.bar(x + width/1.5, adnpn, width/2, label='nonprot-neg') 
plt.ylim(top=1)
ax2.set_ylabel('acc')
ax2.set_title('TPR n TNR AdaFair compass-race')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(prop={"size":7})

rects1 = ax3.bar(x - width, abpp, width/2, label='protpos')
rects2 = ax3.bar(x - width/2, abnpp, width/2, label='nonprot-pos') 
rects3 = ax3.bar(x + width/8, abpn, width/2, label='protneg')
rects4 = ax3.bar(x + width/1.5, abnpn, width/2, label='nonprot-neg') 
plt.ylim(top=1)
ax3.set_ylabel('acc')
ax3.set_title('TPR n TNR AdaBoost compass')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend(prop={"size":7})

rects1 = ax4.bar(x - width, pp, width/2, label='protpos')
rects2 = ax4.bar(x - width/2, npp, width/2, label='nonprot-pos') 
rects3 = ax4.bar(x + width/8, pn, width/2, label='protneg')
rects4 = ax4.bar(x + width/1.5, npn, width/2, label='nonprot-neg') 
plt.ylim(top=1)
ax4.set_ylabel('acc')
ax4.set_title('TPR n TNR MultiFair avg compass')
ax4.set_xticks(x)
ax4.set_xticklabels(labels)
ax4.legend(prop={"size":7})



moddata
pp=[0.52,0.5,0.53]
npp=[0.51,0.52,0.52]
pn=[0.75,0.77,0.77]
npn=[0.75,0.74,0.74]

adpp=[0.74,0.56,0.62]
adnpp=[0.73,0.76,0.75]
adpn=[0.51,0.7,0.7]
adnpn=[0.51,0.46,0.45]

abpp=[0.56,0.43,0.38]
abnpp=[0.37,0.51,0.52]
abpn=[0.73,0.85,0.89]
abnpn=[0.85,0.77,0.75]

ppa=[0.52,0.5,0.51]
nppa=[0.52,0.53,0.52]
pna=[0.74,0.75,0.76]
npna=[0.74,0.73,0.73]
'''