import glob
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import open3d as o3d
import pandas as pd
import trimesh

from scipy.spatial import KDTree

import utils.utils_mp as utils_mp

def eval_reconstruct_gt_pts(rec_mesh_path, gt_mesh_path, name, sample_num=100000):
    # print(rec_mesh_path)
    def normalize_mesh_export(mesh, file_out=None):

        bounds = mesh.extents
        if bounds.min() == 0.0:
            return

        # translate to origin
        translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
        translation = trimesh.transformations.translation_matrix(direction=-translation)
        mesh.apply_transform(translation)

        # scale to unit cube
        scale = 1.0 / bounds.max()
        scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
        mesh.apply_transform(scale_trafo)
        # if file_out is not None:
        #     mesh.export(file_out)
        return mesh

    def get_threshold_percentage(dist, thresholds):
        ''' Evaluates a point cloud.
        Args:
            dist (numpy array): calculated distance
            thresholds (numpy array): threshold values for the F-score calculation
        '''
        in_threshold = [
            (dist <= t).mean() if (dist <= t).any() else 0 for t in thresholds
        ]
        return in_threshold

    def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
        ''' Computes minimal distances of each point in points_src to points_tgt.

        Args:
            points_src (numpy array): source points
            normals_src (numpy array): source normals
            points_tgt (numpy array): target points
            normals_tgt (numpy array): target normals
        '''
        kdtree = KDTree(points_tgt)
        dist, idx = kdtree.query(points_src, workers=-1)

        if normals_src is not None and normals_tgt is not None:
            normals_src = \
                normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
            normals_tgt = \
                normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

            #        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
            #        # Handle normals that point into wrong direction gracefully
            #        # (mostly due to mehtod not caring about this in generation)
            #        normals_dot_product = np.abs(normals_dot_product)

            normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
            normals_dot_product = normals_dot_product.sum(axis=-1)
        else:
            normals_dot_product = np.array(
                [np.nan] * points_src.shape[0], dtype=np.float32)
        return dist, normals_dot_product

    def eval_mesh(pointcloud, pointcloud_tgt,
                  normals=None, normals_tgt=None, thresholds=[0.005]):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        # print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-12)
            for i in range(len(precision))
        ]
        return normals_correctness, chamferL1, chamferL2, F[0]

    try:
        rec_mesh = trimesh.load(rec_mesh_path, process=False)
        if isinstance(rec_mesh, trimesh.Scene):
            rec_mesh = trimesh.load(rec_mesh_path, process=True, force='mesh')
        gt_mesh = trimesh.load(gt_mesh_path, process=False)
        if isinstance(gt_mesh, trimesh.Scene):
            gt_mesh = trimesh.load(gt_mesh_path, process=True, force='mesh')
        # 归一化gt_mesh
        gt_mesh = normalize_mesh_export(gt_mesh, gt_mesh_path)
        # rec_mesh.apply_transform(trans)
        # rec_mesh.apply_transform(scale)
        # rec_mesh.export(rec_mesh_path)
        rec_mesh = normalize_mesh_export(rec_mesh, rec_mesh_path)
    except Exception:
        print(rec_mesh_path)

    # if gt_pts.shape[0] < sample_num:
    #     sample_num = gt_pts.shape[0]
    # sample point for rec

    try:
        pts_rec, idx = rec_mesh.sample(sample_num, return_index=True)
    except Exception as e:
        print(e)
        print(rec_mesh_path, rec_mesh)
    normals_rec = rec_mesh.face_normals[idx]

    pts_gt = None
    normals_gt = None
    if isinstance(gt_mesh, trimesh.PointCloud):
        # print('yes')
        normals_gt = None
        pts_o3d = o3d.io.read_point_cloud(gt_mesh_path)
        normals_gt = np.array(pts_o3d.normals)
        # print(normals_gt.shape)
        pts_gt = np.array(gt_mesh.vertices)
        idx = np.random.choice(pts_gt.shape[0], sample_num, replace=False)
        if normals_gt.shape[0] != 0:
            normals_gt = normals_gt[idx]
        else:
            normals_gt = None
        # print(normals_gt)
        pts_gt = pts_gt[idx]
    else:
        # sample point for gt
        pts_gt, idx = gt_mesh.sample(sample_num, return_index=True)
        normals_gt = gt_mesh.face_normals[idx]
    normals_correctness, chamferL1, chamferL2, f1_mu = eval_mesh(pts_rec, pts_gt, normals_rec, normals_gt)

    out_dict = dict()
    out_dict['name'] = name
    out_dict['normals_correctness'] = normals_correctness
    out_dict['chamferL1'] = chamferL1
    out_dict['chamferL2'] = chamferL2
    out_dict['f1_mu'] = f1_mu
    return out_dict


if __name__ == '__main__':

    shapenet_cat_list = ['airplane','car','bench','rifle','display','telephone','loudspeaker','watercraft','lamp','cabinet','sofa','chair','table']
    method_list = ['digs','ours']

    for n2 in method_list:
        list_e=[]
        for n in shapenet_cat_list:
            exp_name = n
            ## recon mesh path ##

            input_root_path = 'K:\\Dropbox\\exp\\ShapeNet\\1000_0.0\\Align\\shapenet_{}'.format(n2)

            ## gt path ##

            gt_path = 'K:\\Dropbox\\exp\\ShapeNet\\gt\\{}\\'.format(exp_name)

            num_processes = 32

            name = n
            if name in ['input', 'input_n', 'gt', 'Abl_0001_res']:
                continue
            if 'all' in name:
                continue
            input_path = os.path.join(input_root_path, name)
            if not os.path.isdir(input_path):
                continue
            if len(os.listdir(input_path)) == 0:
                continue
            call_params = []
            print(name)
            for i, f in enumerate(os.listdir(input_path)):
                if os.path.splitext(f)[1] not in ['.ply', '.obj', '.off']:
                    continue
                pred_mesh_name = os.path.join(input_path, f)
                gt_name = glob.glob(os.path.join(gt_path, f.split('.')[0] + '*'))[0]
                call_params.append((pred_mesh_name, gt_name, f.split('.')[0]))

            eval_dicts = utils_mp.start_process_pool(eval_reconstruct_gt_pts, call_params, num_processes)
            list_e.append(eval_dicts)

        first = {'name':None,'normals_correctness':None,'chamferL1':None,'chamferL2':None,'f1_mu':None}
        for i in range(13):
            l = list_e[i]
            for j in range(len(l)):
                result = first.copy()
                for key, value in l[j].items():
                    if key in result:
                        if isinstance(result[key], list):
                            result[key].append(value)
                        else:
                            result[key] = [result[key], value]
                    else:
                        result[key] = value

                first =result


        out_file_class = os.path.join(input_root_path, f'eval_meshes_all.csv')
        eval_dicts = first
        eval_df = pd.DataFrame(eval_dicts)
        eval_df = eval_df.sort_values(by=['name'], ignore_index=True)
        mean_se = eval_df.mean()
        mean_se['name'] = 'mean'
        std_se = eval_df.std()
        std_se['name'] = 'std'
        eval_df = eval_df.append(mean_se, ignore_index=True)
        eval_df = eval_df.append(std_se, ignore_index=True)
        eval_df.to_csv(out_file_class)



