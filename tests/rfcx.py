import os
os.sys.path.append('../')
import soundkit
import soundkit.dataprep.rfcx

# Now you can load data from Arbimon
print('Initialize loader')
arbimonloader = soundkit.dataprep.rfcx.ArbimonLoader(config_path='test_config.yaml')

# Load PM metadata into a dataframe
pm_ids = [16817]
print('Get Pattern Matching metadata')
dfp = arbimonloader.get_meta(pm_ids=pm_ids, valid=1)

# Extract wav samples from Arbimon files
arbimonloader.extract_samples(outdir='./tmp/positives/',
                              meta=dfp[:10]) # using a slice of the df for example
