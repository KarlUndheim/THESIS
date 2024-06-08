"""
@File           : mdl_featCalc.py
@Author         : Gefei Kong
@Time:          : 22.11.2023 20:23
------------------------------------------------------------------------------------------------------------------------
@Description    : as below

"""

import os

import numpy as np
import torch

from scipy.stats import skew, kurtosis

def calcFeatures(pts_ch):
    # each channel's features

    if pts_ch.shape[0]==0: # empty file
        # height feature
        hrange, hmax = 0,0
        # intensity feature
        in_max = 0
        in_min = 0
        in_mean= 0
        in_sk  = 0
        in_kut = 0
        in_p90 = 0

        in_std = 0
        in_range = 0
        in_p5 = 0
        in_p10 = 0
        in_p20 = 0
        in_p30 = 0
        in_p40 = 0
        in_median = 0
        in_p60 = 0
        in_p70 = 0
        in_p80 = 0
        in_hcorr = 0
        in_weightedmean = 0
        in_bin1mean = 0
        in_bin2mean = 0
        in_bin3mean = 0
        profiles = [0,0,0]

    else:
        # handle when only 1 point 
        if len(pts_ch.shape)<2:
            # height feature
            hrange, hmax = 0, pts_ch[2]

            # intensity data
            pts_intensity = np.asarray([pts_ch[3]])
            in_sk  = 0.0
            in_kut = 0.0
            in_std = 0.0

        else:
            # height feature
            hrange, hmax = max(pts_ch[:, 2]) - min(pts_ch[:,2]), max(pts_ch[:, 2])

            pts_intensity = pts_ch[:, 3]
            in_sk  = skew(pts_intensity, bias=True)
            in_kut = kurtosis(pts_intensity, bias=True)
            in_std = np.std(pts_intensity) 


        in_max = max(pts_intensity)
        in_min = min(pts_intensity)
        in_mean= np.mean(pts_intensity)
        in_p90 = np.percentile(pts_intensity, 90)
        in_range = in_max - in_min

        in_p5 = np.percentile(pts_intensity, 5)
        in_p10 = np.percentile(pts_intensity, 10)
        in_p20 = np.percentile(pts_intensity, 20)
        in_p30 = np.percentile(pts_intensity, 30)
        in_p40 = np.percentile(pts_intensity, 40)
        in_median = np.percentile(pts_intensity, 50)
        in_p60 = np.percentile(pts_intensity, 60)
        in_p70 = np.percentile(pts_intensity, 70)
        in_p80 = np.percentile(pts_intensity, 80)

        heights = pts_ch[:, 2]
        in_hcorr = np.corrcoef(pts_intensity, heights)

        # Correlation between intensity and height
        in_hcorr = np.corrcoef(pts_intensity, heights)[0, 1] if len(pts_intensity) > 1 else 0
        weighted_mean = np.average(heights, weights=pts_intensity)

        num_bins = 3
        bins = np.linspace(np.min(heights), np.max(heights), num_bins + 1)
        bin_indices = np.digitize(heights, bins) - 1

        profiles = []
        for i in range(num_bins):
            bin_intensities = pts_intensity[bin_indices == i]
            if len(bin_intensities) > 0:
                bin_stats = [np.mean(bin_intensities)]
            else:
                bin_stats = [0]
            profiles.extend(bin_stats)
        

    feats_h = [hrange, hmax]
    feats_inten = [in_max, in_min, in_mean, in_sk, in_kut, in_std, in_range, in_p5, in_p10, in_p20, in_p30, in_p40, in_median, in_p60, in_p70, in_p80, in_p90, in_hcorr] + profiles


    return feats_h, feats_inten # list, len=8


# KARL: Improved calcfeatures
def calcFeaturesImproved(pts_ch):

    # 'hcorr','weighted_mean','bin1mean','bin2mean','bin3mean'
    intensity_features = ['min','mean','max','sk','kut','p90','p80','p70','p60','p50','p40','p30','p20','p10','p5','std','range'] + ['bin']*48

    # Initialize features
    # handle the empty files
    if pts_ch.shape[0] == 0:
        return ([0, 0], [0] * len(intensity_features))  # Adjust the number of zeros based on features

    # only 1 point
    if len(pts_ch.shape)<2: 
        # height feature
        heights = np.asarray([pts_ch[2]])

        # intensity data
        pts_intensity = np.asarray([pts_ch[3]])

    else:
        pts_intensity = pts_ch[:, 3]
        heights = pts_ch[:, 2]

    # Features from reference paper
    in_max = np.max(pts_intensity) 
    in_min = np.min(pts_intensity) 
    in_mean = np.mean(pts_intensity)
    in_std = np.std(pts_intensity)
    in_range = in_max - in_min 
    in_sk = skew(pts_intensity, bias=True) if len(pts_intensity) > 1 else 0
    in_kut = kurtosis(pts_intensity, bias=True) if len(pts_intensity) > 1 else 0
    percentiles = np.percentile(pts_intensity, [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]) if len(pts_intensity) > 1 else np.zeros(10)


    # BIN features
    profiles = []
    b = [2,3,4,5,10]

    for num_bins in b:
        # these are the heights of the edges of the bins
        bin_edges = np.linspace(np.min(heights), np.max(heights), num_bins + 1)

        # Allocate each height to its bin
        height_to_bin = np.digitize(heights, bin_edges) - 1

        for i in range(num_bins):
            # For all the heights in this bin: match with the corresponding intensity
            intensities_to_bin = pts_intensity[height_to_bin == i]

            # to avoid error:
            if len(intensities_to_bin) > 0:
                bin_feats = [np.mean(intensities_to_bin), np.max(intensities_to_bin)]
            else:
                bin_feats = [0,0]
            profiles.extend(bin_feats)
    

    # Height features
    hmax = np.max(heights) if len(heights) > 0 else 0
    hmin = np.min(heights) if len(heights) > 0 else 0
    hrange = hmax-hmin

    feats_h = [hrange, hmax]
    feats_inten = [in_max, in_min, in_mean, in_sk, in_kut, in_std, in_range] + list(percentiles) + profiles

    return feats_h, feats_inten


