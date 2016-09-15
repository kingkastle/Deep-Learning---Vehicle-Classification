# -*- coding: utf-8 -*-
'''
Created on Aug 19, 2016

@author: rafaelcastillo


This script is intended to download pictures from Imagenet (http://image-net.org/) 
based on the synset by HTTP request as here described: 

        http://image-net.org/download-imageurls
        
This project focus on the flowers classification (http://image-net.org/explore?wnid=n11669921), 
corresponding "WordNet ID" (wnid) of a synset of the different flowers families
are stored in local file: 

        'flowers_synsets.txt' 
        
The different wnid included in such files are used to download pictures for each of the flower families:

        http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=[wnid]

'''

import urllib2
import os
import numpy as np
import logging





def download(url,local_dataset,file_name):
    '''
    This function is employed to download files by HTTP
    
    Args:
        * url: url of picture to download
        * local_dataset: root path to locate pictures
        * file_name: name of picture name in local folder
        
    Return:
        1: picture downloaded successfully
        0: picture not downloaded
    
    '''
    try:
        furl = urllib2.urlopen(url)
        finalurl = furl.geturl() # Since some pictures are no longer available, finalurl is used to detect url redirection:
        if url != finalurl: 
            logging.info('File no longer available: {0}'.format(url))
            return 0
        wnid = file_name.split("_")[0]
        local_path = local_dataset + "/" + wnid
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        f = file("{0}.jpg".format(local_path + "/" + file_name),'wb')
        f.write(furl.read())
        f.close()
    except:
        logging.info('Unable to download file {0}'.format(url))
        return 0
    return 1

def worker(procnum, process_number,return_list):
    '''
    Worker function to perform multiprocessing
    
    Arg:
        * procnum: list with url, local Dataset directory and file_name elements for download function.
        * return_list: list to store download function status
        
    Return:
        none
    '''
    url,local_dataset,file_name = procnum[0],procnum[1],procnum[2]
    download_status = download(url, local_dataset, file_name)
    return_list[process_number] = download_status
    
def process_jobs(jobs,return_list):
    '''
    Deploy all appended processes in jobs
    
    Arg:
        * jobs: list with appended processes
        * return_list: list with all processes' outputs
    
    Return:
        Sum of all processes' outputs    
    '''
    for p in jobs:
        p.join(2)
        # If thread is active
        if p.is_alive():
            logging.info( "Process is running... let's kill it...")
            # Terminate process
            p.terminate()
    return np.sum(return_list)
    

if __name__ == '__main__':
    print "download module loaded..."
    
    
            
        
        
        
        
    
    

