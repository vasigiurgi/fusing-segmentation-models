#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2


# Read paths of validation split
def read_path(dir_name, file_name):
    dir_file_name=os.path.join(dir_name,file_name)
    f = open(dir_file_name, 'r')
    list_path_names = [line.replace("\n", "") for line in f.readlines()]
    f.close()
    return list_path_names

#zero unpadding
def zero_unpadding(array_img, h_actual, w_actual, img_size):  
    dh=int((img_size[0]-h_actual)/2)
    dw=int((img_size[1]-w_actual)/2)
    zero_unpadded_img=array_img[dh:dh+h_actual,dw:dw+w_actual,:]
    return zero_unpadded_img

# change from cluster path to local path
def clu_to_local(file_path):
    if file_path.startswith ("/environment"):
        file_path=file_path.split("workdir/")
        file_path=os.path.join("/environment/workdir/Documents/", file_path[1])
    return file_path

def getGroundTruth(fileNameGT):
    '''
    Returns the ground truth maps for roadArea and the validArea 
    :param fileNameGT:
    '''
    # Read GT
    assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
    #full_gt = cv2.imread(fileNameGT, cv2.CV_LOAD_IMAGE_UNCHANGED)
    full_gt = cv2.imread(fileNameGT, -1)
    #attention: OpenCV reads in as BGR, so first channel has Blue / road GT
    roadArea =  full_gt[:,:,0] > 0
    validArea = full_gt[:,:,2] > 0

    return roadArea, validArea

def evalExp_dBI(gtBin, cur_evi, validMap = None, validArea=None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin:
    :param cur_evi:
    :param validMap:
    :param validArea:
    '''

    assert len(cur_evi.shape) == 3, 'Wrong size of input evi map'
    assert len(gtBin.shape) == 2, 'Wrong size of input gt map'

    
    #Merge validMap with validArea
    if validMap!=None: # .all() added
        if validArea!=None:
            validMap = (validMap == True) & (validArea == True)
    elif validArea.all()!=None:
        validMap=validArea

    # True Positive
    if validMap.all()!=None:
        P_map = (gtBin == True) & (validMap == True)
        TP = np.sum ((cur_evi[:,:,1] == True) & (P_map == True))
    else:
        TP = np.sum ((cur_evi[:,:,1] == True) & (gtBin == True))
        
    # True Negative        
    if validMap.all()!=None:
        N_map = (gtBin == False) & (validMap == True)
        TN = np.sum ((cur_evi[:,:,0] == True) & (N_map == True))
    else:
        TN = np.sum ((cur_evi[:,:,0] == True) & (gtBin == False))
        
    # gt Positive and Negative
    if validMap.all()!=None:
        posNum = np.sum((gtBin == True) & (validMap == True))
        negNum = np.sum((gtBin == False) & (validMap == True))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    
    # Ignorance on gt Positive
    if validMap.all()!=None:
        P_map = (gtBin == True) & (validMap == True)
        Omega = cur_evi[:,:,2] == True
        OmegaPos = np.sum((Omega == True) & (P_map == True))
    else:
        Omega = cur_evi[:,:,2] == True
        OmegaPos = np.sum ((Omega == True) & (gtBin == True))
    
    # Ignorance on gt Negative
    if validMap.all()!=None:
        N_map = (gtBin == False) & (validMap == True)
        Omega = cur_evi[:,:,2] == True
        OmegaNeg = np.sum ((Omega == True) & (N_map == True))
    else:
        Omega = cur_evi[:,:,2] == True
        OmegaNeg = np.sum ((Omega == True) & (gtBin == False))
    
    FP = negNum - TN - OmegaNeg
    FN = posNum - TP - OmegaPos
    
    return TP, FP, FN, TN, OmegaPos, OmegaNeg 


class dataStructure: 
    '''
    All the defines go in here!
    '''
    
    cats = ['um_road', 'umm_road', 'uu_road']
    calib_end = '.txt'
    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_propertyList = ['TP', 'FP', 'FN', 'TN', 'OmegaPos', 'OmegaNeg', 'ER', 'IR' ] 

#########################################################################
# function that does the evaluation
def mainEval_dBI(result_dir, gt_fileList_all, debug = False):

    '''
    main method of evaluateRoad
    :param result_dir: directory with the result propability maps, e.g., /home/elvis/kitti_road/my_results
    :param debug: debug flag (OPTIONAL)
    '''
    
    print("Starting evaluation ..." ) 
    print ("Available categories are:", dataStructure.cats)
    
    f= open('stage_2_evaluation_metrics.txt', 'w')
    
    assert os.path.isdir(result_dir), 'Cannot find result_dir: %s ' %result_dir
    
    # In the submission_dir we expect the probmaps! 
    submission_dir = result_dir
    assert os.path.isdir(submission_dir), 'Cannot find %s, ' %submission_dir
    
    # init result
    dBI_eval_scores = [] # the eval results in a dict
    eval_cats = [] # saves all categories that were evaluated

    # New, evaluating for urban road 
    ur_totalTP=0
    ur_totalFP=0
    ur_totalFN=0
    ur_totalTN=0
    ur_totalOmegaPos=0
    ur_totalOmegaNeg=0
    
    #New, evaluating for urban road category
    for cat in dataStructure.cats:
        print ("Execute evaluation for category ...", cat)
        #f.write('%s %s:\n' %("Evaluation metrics for category", cat))
        gt_fileList=[]
        for fname in gt_fileList_all:
            fname_key=fname.split('/')[-1]
            if fname_key.startswith(cat) and fname_key.endswith(dataStructure.gt_end):
                gt_fileList.append(fname)

        assert len(gt_fileList)>0, 'Error reading ground truth'
        # Init data for categgory
        category_ok = True # Flag for each cat
        totalTP=0
        totalFP=0
        totalFN=0
        totalTN=0
        totalOmegaPos=0
        totalOmegaNeg=0
        
        for fn_curGt in gt_fileList:
            
            file_key = fn_curGt.split('/')[-1].split('.')[0]
            if debug:
                print ("Processing file: ", file_key)
            
            # Read GT
            cur_gt, validArea = getGroundTruth(fn_curGt)
                        
            # Read probmap and normalize
            fn_curEvi = os.path.join(submission_dir, file_key + dataStructure.prob_end)
            
            if not os.path.isfile(fn_curEvi):
                print ("Cannot find file: %s for category %s." %(file_key, cat))
                print ("--> Will now abort evaluation for this particular category.")
                category_ok = False
                break
            
            cur_evi_im = cv2.imread(fn_curEvi,-1)
            #cur_prob = np.clip( (cur_prob.astype('f4'))/(np.iinfo(cur_prob.dtype).max),0.,1.)  # check the max of dtype
            cur_evi = cur_evi_im>0
            
            TP, FP, FN, TN, OmegaPos, OmegaNeg  = evalExp_dBI(cur_gt, cur_evi, validMap = None, validArea=validArea)
            
            #assert FN.max()<=posNum, 'BUG @ poitive samples'
            #assert FP.max()<=negNum, 'BUG @ negative samples'
            
            assert TP>=0, 'BUG @ TP negative'
            assert FP>=0, 'BUG @ FP negative'
            assert FN>=0, 'BUG @ FN negative'
            assert TN>=0, 'BUG @ TN negative'
            
            
            # collect results for whole category
            totalTP += TP
            totalFP += FP
            totalFN += FN
            totalTN += TN
            totalOmegaPos+=OmegaPos
            totalOmegaNeg+=OmegaNeg
                    
        if category_ok:
            totalPixelCount = totalTP + totalFP + totalFN + totalTN + totalOmegaPos + totalOmegaNeg
            ErrorRate = (totalFP + totalFN)*100/totalPixelCount
            IgnoranceRate = (totalOmegaPos + totalOmegaNeg)*100/totalPixelCount
            pred_dic = {}
            pred_dic ={'TP':totalTP, 'FP':totalFP, 'FN':totalFN, 'TN':totalTN, 'OmegaPos':totalOmegaPos, 'OmegaNeg':totalOmegaNeg, 'ER':ErrorRate, 'IR':IgnoranceRate}
            print ("Computing evaluation scores...")
            # Compute eval scores!
            dBI_eval_scores.append(pred_dic)
            eval_cats.append(cat)
            
            for property in dataStructure.eval_propertyList:
                pass
                #f.write('%s: %s\n' %(property, dBI_eval_scores[-1][property]))
            print ("Finished evaluating category: %s " %(eval_cats[-1],)) 
        
        #New, evaluation for urban road category
        ur_totalTP+=totalTP
        ur_totalFP+=totalFP
        ur_totalFN+=totalFN
        ur_totalTN+=totalTN
        ur_totalOmegaPos+=totalOmegaPos
        ur_totalOmegaNeg+=totalOmegaNeg
    print ("Execute evaluation for category urban road ..." )
    ur_totalPixelCount = ur_totalTP + ur_totalFP + ur_totalFN + ur_totalTN + ur_totalOmegaPos + ur_totalOmegaNeg
    ur_ErrorRate = (ur_totalFP + ur_totalFN)*100/ur_totalPixelCount
    ur_IgnoranceRate = (ur_totalOmegaPos + ur_totalOmegaNeg)*100/ur_totalPixelCount
    f.write('%s\n' %("Evaluation metrics for category urban road (comulative):"))
    ur_dBI_eval_scores={}
    ur_dBI_eval_scores={'TP':ur_totalTP, 'FP':ur_totalFP, 'FN':ur_totalFN, 'TN':ur_totalTN, 'OmegaPos':ur_totalOmegaPos, 'OmegaNeg':ur_totalOmegaNeg, 'ER':ur_ErrorRate, 'IR':ur_IgnoranceRate}
    
    for property in dataStructure.eval_propertyList:
        f.write('%s: %s\n' %(property, ur_dBI_eval_scores[property]))
    f.close()
    print ("Finished evaluation category: urban road")

    # New, evaluating for urban road category
    
    if len(eval_cats)>0:     
        print ("Successfully finished evaluation for %d categories: %s " %(len(eval_cats),eval_cats))
        return True
    else:
        print ("No categories have been evaluated!")
   
    return False
    

    
