import os
import shutil
import tempfile
import yaml
import boto3
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import multiprocessing as mp
import sqlalchemy as db
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
from soundkit.dataprep.audio import convert_to_wav
from soundkit.config import get_config


class ArbimonLoader:
    def __init__(self, config_path=None):
        
        self.meta = None
        self._config = get_config(config_path)
        _secret = self._config['secret']
        # mysql
        engine   = create_engine('mysql+mysqlconnector://' + \
                                 _secret['mysql_user'] + ':' + \
                                 _secret['mysql_pass'] + '@' + \
                                 _secret['mysql_host'] + ':' + \
                                 str(_secret['mysql_port']) + '/' + \
                                 _secret['mysql_schema'])
        metadata = db.MetaData()
        self.Session  = sessionmaker(bind=engine, autocommit=False)
        self.pm_rois = db.Table('pattern_matching_rois',metadata,autoload=True,autoload_with=engine)
        self.recordings = db.Table('recordings',metadata,autoload=True,autoload_with=engine)
        self.species = db.Table('species',metadata,autoload=True,autoload_with=engine)
        
    
    def get_meta(self, pm_ids, valid=1, pm_batch_size = 25):

        session  = self.Session()

        for batch_start in range(0, len(pm_ids), pm_batch_size):
        
            query = db.select([self.pm_rois.c.recording_id,
                               self.pm_rois.c.species_id,
                               self.pm_rois.c.songtype_id,
                               self.pm_rois.c.x1,
                               self.pm_rois.c.y1,
                               self.pm_rois.c.x2,
                               self.pm_rois.c.y2,
                               self.pm_rois.c.validated,
                               self.pm_rois.c.score]) \
            .where(and_(self.pm_rois.c.pattern_matching_id.in_ \
                        (pm_ids[batch_start:(batch_start+pm_batch_size)]),
                        self.pm_rois.c.validated==valid))
            result = session.execute(query).fetchall()
            
            # Put result in a data frame
            dfbatch = pd.DataFrame(result)
            if dfbatch.shape[1]==0:
                return None
            dfbatch.columns = ['recording_id','species_id','songtype_id','t_min','f_min','t_max','f_max','validated', 'score']

            query = db.select([self.species.c.scientific_name,
                               self.species.c.species_id]).where(self.species.c.species_id.in_(dfbatch.species_id))
            result = session.execute(query).fetchall()
            dtn = dict((v,k) for (k,v) in result) # create a mapping between species name and _id
            dfbatch['species_name'] = [dtn[i] for i in dfbatch['species_id']]

            query = db.select([self.recordings.c.uri,
                               self.recordings.c.recording_id,
                               self.recordings.c.datetime]) \
            .where(self.recordings.c.recording_id.in_(list(set(dfbatch['recording_id']))))
            result = session.execute(query).fetchall()
            dtn = dict((v,k) for (k,v,_) in result) # create a mapping between recording ID and uri
            dfbatch['uri'] = [dtn[x] for x in dfbatch['recording_id']] # add recording_names to meta df
            dtn = dict((v,k) for (_,v,k) in result) # create a mapping between recording ID and datetime
            dfbatch['datetime'] = [dtn[x] for x in dfbatch['recording_id']] # add datetime to meta df
            
            if batch_start==0:
                dfout = dfbatch
            else:
                dfout = pd.concat([dfout, dfbatch])
        
        session.close()
        
        # Remove duplicate detections at 1-second resolution
        dfout['t_max_sec'] = [round(i) for i in dfout.t_max]
        dfout['t_min_sec'] = [round(i) for i in dfout.t_min]
        dfout.drop_duplicates(subset=[i for i in dfout.columns if i not in ['score', 'tod', 'datetime', 't_min', 't_max']], inplace=True)
        del dfout['t_max_sec']
        del dfout['t_min_sec']
        # Reset row indices
        dfout.reset_index(drop=True, inplace=True)
        
        # Add unique identifier for each class
        dfout['class_code'] = [str(i[1]['species_id'])+'_'+str(i[1]['songtype_id']) for i in dfout.iterrows()]
        
        self.meta = dfout
        
        return dfout
    
    
    def extract_samples(self, outdir, overwrite=1, meta=None):
        
        if meta is None:
            meta = self.meta
        
        tmpname = next(tempfile._get_candidate_names())
        tmpdir = './tmp/'+tmpname+'/'
        tmpdir = os.path.expanduser(tmpdir)
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        else:
            shutil.rmtree(tmpdir)
            os.makedirs(tmpdir)
        meta.to_csv(tmpdir+'/meta.csv')
            
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        else:
            if overwrite:
                shutil.rmtree(outdir)
                os.makedirs(outdir)
            
        rec_ids = sorted(list(set(meta.recording_id)))
        pool = mp.Pool(mp.cpu_count())
        pool.map(_extract_samples, [i for i in zip(range(len(rec_ids)), 
                                                   rec_ids,
                                                   (tmpdir,)*len(rec_ids),
                                                   (outdir,)*len(rec_ids),
                                                   (self._config['secret'],)*len(rec_ids),
                                                   ({
                                                       'window_seconds': self._config['fit']['window_seconds'],
                                                       'sample_rate': self._config['audio']['sample_rate'],
                                                       'input_shape': int(round(self._config['fit']['window_seconds'] * self._config['audio']['sample_rate']))
                                                   },)*len(rec_ids),
                                                   (len(rec_ids),)*len(rec_ids))])
        pool.close()
        shutil.rmtree(tmpdir)
        print('')  
           
    
    def get_species_names(self, species_ids):
        
        session = self.Session()
        query = db.select([species.c.species_id,
                           species.c.scientific_name]).where(species.c.species_id.in_(species_ids))
        ids = [i[0] for i in session.execute(query).fetchall()]
        names = [i[1] for i in session.execute(query).fetchall()]
        result = {k: v for k,v in zip(ids, names)}
        session.close()

        return result
        
        
def _extract_samples(inp):
    
    count = inp[0]
    rec_id = inp[1]
    tmpdir = inp[2]
    outdir = inp[3]
    _secret = inp[4]
    _config = inp[5]
    nfiles = inp[6]
    
    # s3
    s3_sieve = boto3.resource('s3',
                              aws_access_key_id=_secret['aws_access_key_id_sieve'],
                              aws_secret_access_key=_secret['aws_secret_access_key_sieve'])
    s3_rfcx = boto3.resource('s3', 
                             aws_access_key_id=_secret['aws_access_key_id_rfcx'],
                             aws_secret_access_key=_secret['aws_secret_access_key_rfcx'])
    
    if count%100==0:
        print(str(count)+'/'+str(nfiles)+' source files')
        
    # get metadata
    roi_data = pd.read_csv(tmpdir+'/meta.csv', index_col=0)
    roi_data.t_min = roi_data.t_min.astype(float)
    roi_data.t_max = roi_data.t_max.astype(float)
    roi_data.f_min = roi_data.f_min.astype(float)
    roi_data.f_max = roi_data.f_max.astype(float)   
    
    rec_loaded = False
    
    tmp = roi_data[roi_data.recording_id==rec_id] # get the subset of annotations for current recording
    uri = tmp.uri.iloc[0]
    
    audio_filename = str(rec_id)+'.'+uri.split('.')[-1]
        
    for c in range(len(tmp)): # loop over annotations for current recording
                
        class_code = tmp.iloc[c].class_code
        if not os.path.exists(outdir+str(class_code)):
            try:
                os.mkdir(outdir+str(class_code))
            except Exception as e:
                if 'exists' in str(e):
                    continue
            
        second_start, second_end = [tmp.iloc[c].t_min, tmp.iloc[c].t_max]
        filename = audio_filename.split('.')[0]+'_'+str(round(second_start,2))+'-'+str(round(second_end,2))+'.wav'
        second_start += ((second_end-second_start) - _config['window_seconds'])/2
        second_start = np.max([0, second_start])
                    
        if not os.path.exists(outdir+str(class_code)+'/'+filename):
            
            # load the recording if needed
            if not rec_loaded:
                try:
                    # download and resample if needed
                    if not os.path.exists(tmpdir+audio_filename):
                        if uri.startswith('project'):
                            s3_sieve.Bucket('arbimon2').download_file(uri, tmpdir+audio_filename)
                        elif uri.startswith('202'):
                            s3_rfcx.Bucket('rfcx-streams-production').download_file(uri, tmpdir+audio_filename)
                        convert_to_wav(tmpdir+audio_filename, tmpdir+audio_filename.split('.')[0]+'.wav', sample_rate=_config['sample_rate'])
                        os.remove(tmpdir+audio_filename)
                        audio_filename = audio_filename.split('.')[0]+'.wav'
                    # load audio
                    y, sr = librosa.load(tmpdir+audio_filename, sr=None)
                    assert sr==_config['sample_rate'], str(i)+' had wrong sample rate: '+str(sr)
                    rec_loaded = True
                except Exception as e:
                    print(e)
                    continue

            # crop the sample
            sample_start = np.min([int(second_start*sr), len(y)-_config['input_shape']])
            sample_end = sample_start+_config['input_shape']
            clip = y[sample_start : sample_end]

            # checks
            if not clip.shape==(_config['input_shape'],):
                print('clip had shape '+str(clip.shape))
                continue

            # save
            sf.write(outdir+str(class_code)+'/'+filename, clip, sr, 'PCM_16')
                                
    if os.path.exists(tmpdir+audio_filename):
        os.remove(tmpdir+audio_filename)    

    
    