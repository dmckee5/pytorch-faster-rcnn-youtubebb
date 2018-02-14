# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Youtube Bounding Box added by Mark Buckler
# --------------------------------------------------------
# This dataset loader expects to read the Youtube Bounding
# Box after being converted into the VOC2007 format. The
# converter and downloader can be found here:
# https://github.com/mbuckler/youtube-bb
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import PIL
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
import multiprocessing
from .voc_eval import voc_eval
from model.config import cfg

class youtubebb(imdb):
    def __init__(self, image_set, year, use_diff=False):
        name = 'youtubebb_' + year + '_' + image_set
        if use_diff:
            name += '_diff'
        imdb.__init__(self, name)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = os.path.join(self._devkit_path, 'youtubebb' + self._year)
        self._classes = ('__background__', # always index 0
                         'person','bird','bicycle','boat','bus',
                         'bear','cow','cat','giraffe','potted plant',
                         'horse','motorcycle','knife','airplane',
                         'skateboard','train','truck','zebra','toilet',
                         'dog','elephant','umbrella','car')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': use_diff,
                       'matlab_eval': False,
                       'rpn_file': None}
        if not self.config['cleanup']:
            print('Results files will be saved with template: %s'
                  % self._get_voc_results_file_template())

        assert os.path.exists(self._devkit_path), \
                'youtubebbdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /youtubebbdevkit2017/youtubebb2017/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where youtubebb is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'youtubebbdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

#        print("image index length: " + str(len(self._image_index)))
        print("Loading ground-truth roidb annotations")
        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2017 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # youtubebbdevkit/results/youtubebb2017/Main/<comp_id>_det_test_person.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'youtubebb' + self._year,
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._devkit_path,
            'youtubebb' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'youtubebb' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')

        num_iou_thresholds = 10
        sum_aps = np.zeros(len(self._classes)-1) # minus 1 for background
        for iou_thresh in np.linspace(.5, 0.95, num_iou_thresholds, endpoint=True):
            print('Using overlap threshold {:.6f}'.format(iou_thresh))
            aps = []
            # The PASCAL VOC metric changed in 2010.
            # The Youtube Bounding Box converter uses the 2007 format, so we
            # establish that here.
            use_07_metric = True
            print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            for i, cls in enumerate(self._classes):
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                rec, prec, ap = voc_eval(
                    filename, annopath, imagesetfile, cls, cachedir, ovthresh=iou_thresh,
                    use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
                aps += [ap]
                print('AP for {} = {:.4f}'.format(cls, ap))
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                    pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            sum_aps += np.asarray(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('')

        # print mAP across iou thresholds
        print('Evaluating AP values averaged across all IoU thresholds')
        iou_avg_aps = sum_aps/float(num_iou_thresholds)
        classes_without_background = [cls for cls in self._classes if cls != '__background__']
        for i, cls in enumerate(classes_without_background):
            print('AP for {} = {:.4f}'.format(cls, iou_avg_aps[i]))
        print('Mean AP = {:.4f}'.format(np.mean(iou_avg_aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in iou_avg_aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(iou_avg_aps)))
        print('~~~~~~~~')

        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


    # use this method to output files with specifics on which detections were missed
    def _do_python_eval_analysis(self, output_dir='output', iou_thresh=0.5):
        annopath = os.path.join(
            self._devkit_path,
            'youtubebb' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'youtubebb' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')

        print('Using overlap threshold {:.6f}'.format(iou_thresh))
        aps = []
        # The PASCAL VOC metric changed in 2010.
        # The Youtube Bounding Box converter uses the 2007 format, so we
        # establish that here.
        import pdb
        pdb.set_trace()

        use_07_metric = True
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap, tpfp_dict = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=iou_thresh,
                use_07_metric=use_07_metric, use_diff=self.config['use_diff'])
            # with open(os.path.join(output_dir, cls+str(iou_thresh).replace('.', '_')
            #         + '_thresh_tpfp_detection_eval.pkl'), 'wb') as f:
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_' + str(iou_thresh).replace('.', '_')
                    + '_thresh_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap, 'tpfp_dict': tpfp_dict}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')

        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')



    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir, do_analysis=False):
        self._write_voc_results_file(all_boxes)
        if do_analysis:  # this is for getting tp/fp results for a single overlap threshold
            self._do_python_eval_analysis(output_dir)
        else:
            self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    # def get_ind_video_width(self, i):
    #     return PIL.Image.open(self.image_path_at(i)).size[0]

    # def _get_widths(self):
    #     num_threads = cfg.PREPROCESSING_THREADS
    #     print("Using thread pool of size " + str(num_threads))
    #     pool = multiprocessing.Pool(num_threads)
    #     return pool.map(self.get_ind_video_width, range(self.num_images))
    #     # return [PIL.Image.open(self.image_path_at(i)).size[0]
    #     #         for i in range(self.num_images)]

    def _get_widths(self):
        sizes = self.get_img_size_dict()
        return [sizes[self.image_path_at(i)][0]  # 0 index of size tuple is width
                for i in range(self.num_images)]

    def get_img_size_dict(self):
        num_threads = cfg.PREPROCESSING_THREADS

        cache_size_dir = os.path.join(self.cache_path, 'image_sizes')
        file_name = self.name + '_sizes.pkl'
        cache_size_file = os.path.join(cache_size_dir, file_name)
        if not os.path.exists(cache_size_file):
            print('Cached image size file not found for ' + self.name)
            if not os.path.exists(cache_size_dir):
                os.makedirs(cache_size_dir)

            print("Using thread pool of size %d to read image dimensions" % num_threads)
            pool = multiprocessing.Pool(num_threads)
            sizes = dict(pool.map(self.get_ind_video_name_size, range(self.num_images)))
            print('Saving image size cache file...')

            with open(cache_size_file, "wb") as f:
                pickle.dump(sizes, f)

        else:
            print('Loading cached image size file for ' + self.name)
            with open(cache_size_file, "rb") as f:
                sizes = pickle.load(f)
        return sizes

    def get_ind_video_name_size(self, i):
        return self.image_path_at(i), PIL.Image.open(self.image_path_at(i)).size


if __name__ == '__main__':
    from datasets.youtubebb import youtubebb

    d = youtubebb('trainval', '2017')
    res = d.roidb
    from IPython import embed;

    embed()
