import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import pyDBoW3 as bow
from scipy import stats
import time
import glob
from sklearn.preprocessing import minmax_scale



CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# mulran or kitti 
DATASET_TYPE = "mulran"

generate_images = False
compare_gt = True
set_up_database = True

ilk = os.path.join(CURRENT_DIR, "intensity_images")
for f in os.listdir(ilk):
    os.remove(os.path.join(ilk, f))

ilk = os.path.join(CURRENT_DIR, "comparison_images")
for f in os.listdir(ilk):
    os.remove(os.path.join(ilk, f))

for f in os.listdir(CURRENT_DIR):
    if ".png" in f:
        os.remove(os.path.join(CURRENT_DIR, f))
    
if set_up_database:


    # setting up database 
    voc = bow.Vocabulary()
    print(dir(voc))
    voc.load(r"C:\Users\Intern\Downloads\ORBvoc0\ORBvoc1\ORBvoc.txt")
    db = bow.Database()
    db.setVocabulary(voc, False, 0)
    print(bow.Vocabulary)



def do_intensity_projection(points, proj_W , proj_H,  proj_fov_up, proj_fov_down, fn, idx):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """

    # print(points.shape)

    points = points[points.any(axis=1)]

    proj_range = np.zeros((proj_H, proj_W),
                              dtype=np.float64)

    # unprojected range (list of depths for each point)
    unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((proj_H, proj_W, 4), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    proj_remission = np.full((proj_H, proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.zeros((proj_H, proj_W),
                              dtype=np.int32)       # [H,W] mask




    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad


   
    depth = np.linalg.norm(points[:,:3], 2, axis=1)

    # print(points[:10,:])
    

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x) 
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    proj_x = np.nan_to_num(proj_x)

    proj_y = np.nan_to_num(proj_y)
    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

   
    

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

    proj_y = np.copy(proj_y)  # stope a copy in original order


    # # copy of depth in original order
    # unproj_range = np.copy(depth)

    # indices = np.arange(depth.shape[0])
    # order = np.argsort(depth)[::-1]
    # depth = depth[order]
    # indices = indices[order]
    # points = points[order]

    # proj_y = proj_y[order]
    # proj_x = proj_x[order]
    

    if  DATASET_TYPE == "kitti":
        intensities = points[:,3]
        print("kitti")
        # intensities = np.minimum(intensities, 1000)
        # i_min = intensities.min()
        # i_max = intensities.max()
        # intensities = (intensities - i_min)/(i_max - i_min)



    if DATASET_TYPE == "mulran" or DATASET_TYPE == "mulran2":
        intensities = points[:,3]
        intensities = np.minimum(intensities, 1000)
        i_min = intensities.min()
        i_max = intensities.max()
  
        intensities = (intensities - i_min)/(i_max - i_min)

    if DATASET_TYPE == "dso":
      
        
      
        intensities = points[:,4]
      

        minval = np.percentile(intensities, 2)
        maxval = np.percentile(intensities, 98)
        intensities = np.clip(intensities, minval, maxval)
        # intensities = np.maximum(intensities, 5000)
        # intensities = np.sqrt(intensities)

     


        
        i_min = intensities.min()
        i_max = intensities.max()

        intensities = (intensities - i_min)/(i_max - i_min)

        


        

    

   
    


    
    
    
    
    
    pixel_tracker = {}
    pc_tracker = {}
    # print(proj_x.shape)
    # print(scan_x.shape)

    
    proj_3d_corres = np.zeros((proj_H, proj_W, 3),
                              dtype=np.float64)

    # print(proj_x[:20])
    # print(proj_y[:70])
    
    
    for i in range(proj_x.shape[0]):
        x_val = proj_x[i]
        y_val = proj_y[i]

       

        if proj_range[y_val, x_val] != 0:
            continue

        
      
        intensity = intensities[i]
        
 
        
        
        proj_range[y_val, x_val] = intensity
    
        proj_3d_corres[y_val,x_val, :] = np.array([scan_x[i], scan_y[i], scan_z[i]])
            


    
    proj_range *= 255


   
    
    
 
    proj_range = np.array(proj_range, dtype=np.uint8)

    
    newPicPath = None


   


    img = Image.fromarray(proj_range, 'L')
    pc_name = fn.split('.')[0]
    newPicPath = os.path.join(CURRENT_DIR, "intensity_images", "mulran_" + (str(idx)) + ".png")
    img.save(newPicPath)


    return newPicPath, proj_3d_corres, proj_range


def orb_descriptor_marker(imgPath, proj_3d_corres, proj_range):
    # rando = r"C:\Users\Intern\Pictures\Screenshots\Screenshot (2).png"
    img = proj_range

    
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=2500)
    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # print(len(kp))
    
    corner_scores = []
    points_3d = []
    points_2d = []
    new_kp = []

    
    new_des = []
    
 
    
    for i, kr in enumerate(kp):

   
        w = round(kr.pt[0])
        h = round(kr.pt[1])
    
        x_val = proj_3d_corres[h,w,0]
        
        y_val = proj_3d_corres[h,w,1]
        z_val = proj_3d_corres[h,w, 2]

        # if z_val == 0:
        #     continue
        points_3d.append(proj_3d_corres[h,w])
        points_2d.append([ x_val/z_val, y_val/z_val])
      
        corner_scores.append(kr.response)
    


    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # print(des.shape)


    # img2 = cv2.drawKeypoints(img,kp,np.array([]),color=(0,255,0), flags=0)
    # lamb = imgPath.split('\\')[-1]
    # lamb = lamb.split('.')[0]
    # newPicPath = os.path.join(CURRENT_DIR, "intensity_images", lamb + "_orb.png")
    # cv2.imwrite(newPicPath, img2)

    points_2d = np.array(points_2d)
    points_3d = np.array(points_3d)

    return np.array(kp), des, corner_scores, points_2d, points_3d

def image_feature_comparison(tracker1, tracker2, diff):

    imgPath1 = tracker1['path']
    imgPath2 = tracker2['path']
    kp1 = tracker1['kp']
    kp2 = tracker2['kp']
    des1 = tracker1['des']
    des2 = tracker2['des']
    cs1 = tracker1['corner_scores']
    cs2 = tracker2['corner_scores']
    points_2d_1 = tracker1['points_2d']
    points_2d_2 = tracker2['points_2d']
    points_3d_1 = tracker1['points_3d']
    points_3d_2 = tracker2['points_3d']

    identity = np.eye(3)



    cs1 = sorted(cs1, reverse=True)
    if len(cs1) >= 500:
        cs_threshold = cs1[499]
        chosen_indices = []
        for j,cs in enumerate(cs1):
            if cs >= cs_threshold:
                chosen_indices.append(j)

        
        des1 = des1[chosen_indices]
        


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # inds = np.argsort(cs1)[::-1][:40]
    # print(inds)
    # des1 = des1[inds,:]
    # des2 = des2[inds,:]

    matches = bf.match(des1, des2)
    # matches = sorted(matches, key = lambda x:x.distance)

    points_3d = []
    points_2d = []
    good_indexes = []

    # distances = []
    # for m in matches:
    #     distances.append(m.distance)
    
    # dist_threshold = min(distances) * 2

    # new_matches = []

    # for m in matches:
    #     if m.distance < dist_threshold:
    #         new_matches.append(m)

    # if len(new_matches) <= 12:
    #     return [], new_matches

    
    new_matches = []

    for j,m in enumerate(matches):
        
        

        pt = points_3d_1[m.queryIdx]
        pixel = points_2d_2[m.trainIdx]

        if not (np.isnan(pixel[0])):
            good_indexes.append(j)

            new_matches.append(m)
        
      


        points_3d.append(pt)
        points_2d.append(pixel)

    # if len(new_new_matches) < 30:
    #     return [], new_new_matches

    points_3d = np.array(points_3d, dtype=np.float32)
    # points_3d = points_3d[good_indexes]
    points_2d = np.array(points_2d, dtype=np.float32)
    # points_2d = points_2d[good_indexes]


    
  
    try:
        val, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, identity, None, reprojectionError = 0.1)

        
    except:
        return [], matches, None
  
        
    if inliers is None:
        inliers = []
        
    else:
        inliers = inliers.flatten().tolist()
        

    # new_inliers = []
    # new_matches = []

    # for j, inli in enumerate(inliers):
    #     if matches[j].distance <= 50:
    #         new_inliers.append(inli)
    #         new_matches.append(matches[j])


    if generate_images:

        img = cv2.imread(imgPath1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(imgPath2, cv2.IMREAD_GRAYSCALE)
        
        matching_result = draw_matches(img, kp1, img2, kp2, matches, inliers, None)
        newPicPath = os.path.join(CURRENT_DIR, f"diff_{str(diff)}_matching_inliers.png")
        cv2.imwrite(newPicPath, matching_result)

        matching_result = draw_matches(img, kp1, img2, kp2, matches,None, None)
        newPicPath = os.path.join(CURRENT_DIR, f"diff_{str(diff)}_matching_outliers.png")
        cv2.imwrite(newPicPath, matching_result)

        matching_result = draw_matches(img, kp1, img2, kp2, matches,None, None, filter_by_dist=False)
        newPicPath = os.path.join(CURRENT_DIR, f"diff_{str(diff)}_matching_outliers__no_dist_filter.png")
        cv2.imwrite(newPicPath, matching_result)

        matching_result = cv2.drawMatches(img, kp1, img2, kp2, matches,None,  flags=2)
        newPicPath = os.path.join(CURRENT_DIR, "matching2.png")
        cv2.imwrite(newPicPath, matching_result)

    return inliers, matches, tvec

def draw_matches(img1, kp1, img2, kp2, matches, inliers, ignore_indexes, filter_by_dist=True, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.



    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    if len(img1.shape) == 3:
        new_shape = (img1.shape[0] + img2.shape[0], max(img1.shape[1], img2.shape[1]), img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (img1.shape[0] + img2.shape[0], max(img1.shape[1], img2.shape[1]))
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[img1.shape[0]:img1.shape[0]+img2.shape[0],0:img1.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 1
    thickness = 1
    if color:
        c = color

    # print(new_img.shape)
    distances = []
    for m in matches:
        distances.append(m.distance)
    
    dist_threshold = min(distances) * 2
    # print(dist_threshold)
    
    for i, m in enumerate(matches):
        if inliers:
            if not i in inliers:
                continue
            if ignore_indexes:
                if i in ignore_indexes:
                    continue
        if filter_by_dist:
            if m.distance > 50:
                continue
    
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = tuple(np.random.randint(0,256,3)) if len(img1.shape) == 3 else np.random.randint(0,256)
            c = ( int (c [ 0 ]), int (c [ 1 ]), int (c [ 2 ])) 
        
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.

        try:
            end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
            end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([ 0, img1.shape[0]]))
            cv2.line(new_img, end1, end2, c, thickness)
            cv2.circle(new_img, end1, r, c, thickness)
            cv2.circle(new_img, end2, r, c, thickness)
        except:
            continue
       

    return new_img


if DATASET_TYPE == 'mulran':
    PC_FOLDER = r"C:\Users\Intern\loop_closure\mulran_pointclouds"
    fov_down = -16.5
    fov_up = 16.5
    img_W = 1024
    img_H = 64



    gt= np.fromfile(os.path.join(CURRENT_DIR, "bin_positions.bin"), dtype=np.float32).reshape(-1,3)
    
    compare_gt = True

    scan_dim = 4

if DATASET_TYPE == 'mulran2':
    PC_FOLDER = r"C:\Users\Intern\loop_closure\mulran_riverside"
    fov_down = -16.5
    fov_up = 16.5
    img_W = 1024
    img_H = 64



    gt= np.fromfile(os.path.join(CURRENT_DIR, "mulran_riverside_bin_positions.bin"), dtype=np.float32).reshape(-1,3)
    
    compare_gt = True

    scan_dim = 4


elif DATASET_TYPE == 'dso':
    PC_FOLDER = r"C:\Users\Intern\loop_closure\Ouster Data 2"
    fov_down = -16.5
    fov_up = 16.5
    img_W = 1024
    img_H = 64


    scan_dim = 6

    raw_poses = np.genfromtxt(os.path.join(PARENT_DIR, "dso_poses", "GlobalPoseUTM.csv"), delimiter=",", skip_header=True)
    pcs_timestamps = [a.replace(".bin", "") for a in os.listdir(PC_FOLDER)]
    pcs_timestamps = [a.split("_") for a in pcs_timestamps]
    pcs_timestamps = [(int(a[0]) + int(a[1]))/2 for a in pcs_timestamps]
    print(len(pcs_timestamps))

    poses_timestamps = raw_poses[:,1]

    pc_idx = 0
    pose_idx = 0
    pc_tracker = np.zeros((len(pcs_timestamps), 3))
    pc_tracker[:,0] = pcs_timestamps

    print(poses_timestamps[-5:])
    min_pose_timestamp = min(poses_timestamps)

    lowest_timestamp_read = False


    while True:

    
        if pose_idx >= len(poses_timestamps) - 1 or pc_idx > len(pcs_timestamps) - 1:
            break

        pc_ts = pcs_timestamps[pc_idx]
        current_pose_ts = poses_timestamps[pose_idx]

        if pc_ts < min_pose_timestamp:
            pc_idx += 1
            continue

        if pc_ts == current_pose_ts:
            pc_tracker[pc_idx,1:] = raw_poses[pc_idx,2:]

        if pose_idx >= len(poses_timestamps) - 2:
            break

        next_pose_ts = poses_timestamps[pose_idx + 1]

        if pc_ts > current_pose_ts and pc_ts < next_pose_ts:
            print(pc_idx)
            

            
            interpolation_factor = (pc_ts - current_pose_ts)/(next_pose_ts - current_pose_ts)
            
            current_coords = raw_poses[pose_idx, 2:]
            next_coords = raw_poses[pose_idx +1, 2:]
            new_coords = next_coords * interpolation_factor + current_coords * (1-interpolation_factor)
            pc_tracker[pc_idx,1:] = new_coords
            pc_idx += 1
            continue

        pose_idx += 1

    print(pc_tracker)
    print(f"Lowest timestamp read: {pc_idx}")








    


 

elif DATASET_TYPE == 'kitti':
    PC_FOLDER = r"C:\Users\Intern\loop_closure\pointclouds"
    fov_down = -25.0
    fov_up = 3.0
    img_W = 1024
    img_H = 64

    scan_dim = 4


all_descs = []
all_kps = []
all_paths = []

vocab = []


tracker = []
idx = 0
numClosures = 0
correctClosures = 0
time_tracker = 0
totalPoseError = 0

for j, fn in enumerate(os.listdir(PC_FOLDER)):
   
    if '.bin' in fn:

        closureFlag = False

        # if j % 10 != 0:
        #     continue
        
        print(f"This is idx {idx}")
        start = time.time()
        bin_name = os.path.join(PC_FOLDER, fn)
     
        test_pc = np.fromfile(bin_name, dtype=np.float32).reshape(-1,scan_dim)
        intensities = test_pc[:,3]
        # minval = np.percentile(intensities, 10)
        # maxval = np.percentile(intensities, 95)
        # # selected_bin_points = test_pc[:,:3].any(axis=1)
        
        # # minval = np.percentile(intensities, 0)
        # # maxval = np.percentile(intensities,90)
        # # print(minval)
        # # print(maxval)
        # # cara = 40
        # selected_bin_points = (test_pc[:,5] < maxval)
        # # selected_bin_points = (test_pc[:,0] < cara) & (test_pc[:,0] > -cara) & (test_pc[:,1] < cara) & (test_pc[:,1] > -cara) & (test_pc[:,2] < cara) & (test_pc[:,2] > -cara)
        # # print(test_pc[test_pc[:,3] > 60000])
        
        # test_pc = test_pc[selected_bin_points]

        # print(np.amax(test_pc[:,3:], axis=0))


        # print(test_pc[:10,:])
        # print(test_pc.shape)
        # intensities = test_pc[:,3]
        # distribution = plt.hist(intensities)
        # plt.savefig("distribution.png")

        # test_pc = np.nan_to_num(test_pc)

        # intensities = test_pc[:,3]
        # x_vals = test_pc[:,0]


        # print(intensities.min())
        # print(intensities.max())

        # print(x_vals.min())
        # print(x_vals.max())
     

        picPath, proj_3d_corres, proj_range = do_intensity_projection(test_pc, img_W, img_H, fov_up, fov_down, fn, idx)
      
        kp, des, corner_scores, points_2d, points_3d = orb_descriptor_marker(picPath, proj_3d_corres, proj_range)

        track = {}

        track['path'] = picPath
        track['kp'] = kp
        track['des'] = des
        track['corner_scores'] = corner_scores
        track['points_2d'] = points_2d
        track['points_3d'] = points_3d
      


        tracker.append(track)

        db.add(des)
   
        results = db.query(des, 10, -1)
        
        for r in results:
            if abs(idx - r.Id) > 400  and r.Score >= 0.03:
                inliers, matches, tvec = image_feature_comparison(tracker[r.Id], tracker[idx], idx - r.Id)
                print(len(inliers))
                
                if len(inliers) > 60:
                    print(f"Id {r.Id} has {len(inliers)} inliers")
                    print(f"DBow Score: {r.Score}")
                    print(f"Tvec:")
                    print(tvec)

                    tracker1 = tracker[r.Id]
                    tracker2 = tracker[idx]

                    imgPath1 = tracker1['path']
                    imgPath2 = tracker2['path']

                    kp1 = tracker1['kp']
                    kp2 = tracker2['kp']
                    des1 = tracker1['des']
                    des2 = tracker2['des']

                    img = cv2.imread(imgPath1, cv2.IMREAD_GRAYSCALE)
                    img2 = cv2.imread(imgPath2, cv2.IMREAD_GRAYSCALE)
                    
                    matching_result = draw_matches(img, kp1, img2, kp2, matches, inliers, None)
                    newPicPath = os.path.join(CURRENT_DIR, "comparison_images", f"{str(r.Id)}_{str(idx)}_inliers.png")
                    cv2.imwrite(newPicPath, matching_result)

     
                    closureFlag = True

                    if compare_gt and (DATASET_TYPE == 'mulran' or DATASET_TYPE == 'mulran2'):

                        loc1 = gt[r.Id , :]
                        loc2 = gt[idx, :]

                        gt_dist_vec = loc2 - loc1
                        tvec[2] = 0 
                        gt_dist_vec[2] = 0 
                        dist_vec_error = np.sum(np.abs(tvec - gt_dist_vec))


                    
                        dist = np.sqrt(np.sum(np.square(loc1 - loc2)))

                        print(gt_dist_vec)
                        
                        print(f"Distance between scans: {dist}")
                        print("")

                        print(f"Pose error:{dist_vec_error}")
                        totalPoseError += dist_vec_error

                        if dist < 2:
                            correctClosures += 1

                    if compare_gt and DATASET_TYPE == 'dso':

                        loc1 = pc_tracker[r.Id, 1:]
                        loc2 = pc_tracker[idx, 1:]
                        print(loc1)
                        print(loc2)
                        if loc1.any() and loc2.any():
                            dist = np.sqrt(np.sum(np.square(loc1 - loc2)))

                            print(f"Distance between scans: {dist}")
                            print("")

                            if dist < 2:
                                correctClosures += 1

                    
                    
                    numClosures += 1
                    




        if closureFlag:
            pass

        tt = time.time() - start

        print("Time taken: {}s".format(tt))
        time_tracker += tt
        idx += 1

print(f"Number of closures: {numClosures}")
print(f"Average time spent: {time_tracker/j}")

if compare_gt:
    print(f"Closure accuracy: {correctClosures/numClosures}")
    print(f"Average pose error:{totalPoseError/numClosures}")


