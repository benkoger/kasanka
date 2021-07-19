import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import signal
from sklearn.neighbors import KernelDensity
import copy
import os
import utm
import rasterio
from CountLine import CountLine

import sys
sys.path.append('/home/golden/general-detection/functions')
import koger_tracking as ktf


def mark_bats_on_image(image_raw, centers, radii=None, 
                       scale_circle_size=5, contours=None, 
                       draw_contours=False):
    
    '''
    Draw a bunch of circles on given image
    
    image: 2D or 3D image
    centers: shape(n,2) array of  circle centers
    radii: list of circle radii
    
    '''
    
    if len(image_raw.shape) < 2:
        print('image has too few dimensions')
        return None
    
    if len(image_raw.shape) == 2:
            color = 200
    else:
        if image_raw.shape[2] == 3:
            color = (0, 255, 255)
        else:
            print('image is the wrong shape')
            return None
        
    image = np.copy(image_raw)
    
    if radii is None:
        radii = np.ones(len(centers))
    
    for circle_ind, radius in enumerate(radii):
       
        cv2.circle(image, 
                   (centers[circle_ind, 0].astype(int), 
                    centers[circle_ind, 1].astype(int)), 
                   int(radius * scale_circle_size), color , 1) 
    if draw_contours and contours:
        for contour in contours:
            if len(contour.shape) > 1:

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_d = np.int0(box)
                cv2.drawContours(image, [box_d], 0, (0,255,100), 1)
    return image

def get_tracks_in_frame(frame_ind, track_list):
    """ Return list of all tracks present in frame ind. """
    
    tracks_in_frame = []
    for track in track_list:
        if (track['last_frame'] >= frame_ind 
                and track['first_frame'] <= frame_ind):
            tracks_in_frame.append(track)
    return tracks_in_frame
    
    


def draw_tracks_on_frame(frame, frame_ind, track_list, 
                         positions=None, figure_scale=60, 
                         track_width=2, position_alpha=.5, 
                         draw_whole_track=False, shift=0):
    
    """ Draw all active tracks and all detected bat locations on given frame.
    
    frame: loaded image - np array
    frame_ind: frame number
    track_list: list of all tracks in observation
    positions: all detected bat positions in observation
    figure_scale: how big to display output image
    track_width: width of plotted tracks
    position_alpha: alpha of position dots
    draw_whole_track: Boolean draw track in the future of frame_ind
    shift: compensate for lack of padding in network when drawing tracks
        on input frames
    """


    plt.figure(
        figsize = (int(frame.shape[1] / figure_scale), 
                   int(frame.shape[0] / figure_scale)))
    plt.imshow(frame)
    
    num_tracks = 0
    
    for track in track_list:
        if (track['last_frame'] >= frame_ind 
            and track['first_frame'] <= frame_ind):
            rel_frame = frame_ind - track['first_frame']
            if draw_whole_track:
                plt.plot(track['track'][:, 0] + shift, 
                         track['track'][:, 1] + shift, 
                         linewidth=track_width)
            else:
                plt.plot(track['track'][:rel_frame, 0] + shift, 
                         track['track'][:rel_frame, 1] + shift, 
                         linewidth=track_width)
            
            num_tracks += 1
    if positions:
        plt.scatter(positions[frame_ind][:,0] + shift, 
                    positions[frame_ind][:,1] + shift, 
                    c='red', alpha=position_alpha)
        plt.title('Tracks: {}, Bats: {}'.format(num_tracks, 
                                                len(positions[frame_ind])))
    
def subtract_background(images, image_ind, background_sum):
    '''
    Subtract an averaged background from the image. Average over frame_range in the past and future
    
    images: 3d numpy array (num images, height, width)
    image_ind: index in circular image array
    background_sum: sum of blue channel pixels across 0 dimension of images
    '''
    
    background = np.floor_divide(background_sum, images.shape[0])

    
    # The order of subtraction means dark bats are now light in image_dif
    image_dif =  background - images[image_ind, :, :, 2]
    
    
    
    return image_dif, background


def preprocess_to_binary(image, binary_thresh, background):
    
    '''
    Converts 2D image to binary after rescaling pixel intensity 
    
    image: 2D np array
    low_pix_value: pixel value below which all pixels are set to 0
    high_pix_value: pixel value above which all pixels are set to 255
    binary_thresh: number from 0 - 255, above set to 255, bellow, set to 0
    background: background image (2D probably blue channel)
    '''
    
    
    
#     # Rescale image pixels within range
#     image_rescale = exposure.rescale_intensity(
#         image, in_range=(low_pix_value, high_pix_value), out_range=(0, 255))

    image_rescale = image
    
    # Binarize image based on threshold
    min_difference = 5
    threshold = binary_thresh * background
    threshold = np.where(threshold < min_difference, min_difference, threshold)
    
    binary_image = np.where(image < threshold, 0, 255)
    
    return binary_image

def get_blob_info(binary_image, background=None, size_threshold=0):
    
    '''
    Get contours from binary image. Then find center and average radius of each contour
    
    binary_image: 2D image
    background: 2D array used to see locally how dark the background is
    size_threshold: radius above which blob is considered real
    '''
    
    contours, hierarchy = cv2.findContours(binary_image.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    # Size of bounding rectangles
    sizes = []
    areas = []
    # angle of bounding rectangle
    angles = []
    rects = []
    good_contours = []
    contours = [np.squeeze(contour) for contour in contours]
    
    for contour_ind, contour in enumerate(contours):
        
        
        
        if len(contour.shape) >  1:
            
            rect = cv2.minAreaRect(contour)
            
            if background is not None:
                darkness = background[int(rect[0][1]), int(rect[0][0])]
                if darkness < 30:
                    dark_size_threshold = size_threshold + 22
                elif darkness < 50:
                    dark_size_threshold = size_threshold + 15
                elif darkness < 80:
                    dark_size_threshold = size_threshold + 10
                elif darkness < 100:
                    dark_size_threshold = size_threshold + 5
    #             elif darkness < 130:
    #                 dark_size_threshold = size_threshold + 3
                else:
                    dark_size_threshold = size_threshold
            else:
                dark_size_threshold = 0 # just used in if statement

            area = rect[1][0] * rect[1][1]
            
            if (area >= dark_size_threshold) or background is None:
                centers.append(rect[0])
                sizes.append(rect[1])
                angles.append(rect[2])
                good_contours.append(contour)
                areas.append(area)
                rects.append(rect)
    if centers:
        centers = np.stack(centers, 0)
        sizes = np.stack(sizes, 0)
    else:
        centers = np.zeros((0,2))
            
    return (centers, np.array(areas), good_contours, angles, sizes, rects)

def draw_circles_on_image(image, centers, sizes, rects=None):
    
    '''
    Draw a bunch of circles on given image
    
    image: 2D or 3D image
    centers: shape(n,2) array of  circle centers
    rects: list of minimum bounding rectangles
    
    '''
    
    if len(image.shape) < 2:
        print('image has too few dimensions')
        return None
    
    if len(image.shape) == 2:
            color = 200
            rect_color = 100
    else:
        if image.shape[2] == 3:
            color = (0, 255, 255)
            rect_color = (0,255,100)
        else:
            print('image is the wrong shape')
            return None
    
    for circle_ind, size in enumerate(sizes):       
        cv2.circle(image, (centers[circle_ind, 0].astype(int), centers[circle_ind, 1].astype(int)), 
                   int(np.max(size)), color , 1) 
    if rects:
        for rect in rects:
            box = cv2.boxPoints(rect)
            box_d = np.int0(box)
            cv2.drawContours(image, [box_d], 0, rect_color, 1)
    return image

def update_circular_image_array(images, image_ind, image_files, frame_num, background_sum):
    """ Add new image if nessesary and increment image_ind.
    
    Also update sum of pixels across array for background subtraction.
    
    If frame_num is less than half size of array than don't need to 
    replace image since intitally all images in average are in the future.
    
    images: image array size (num images averaging, height, width, channel)
    image_ind: index of focal frame in images
    image_files: list of all image files in observation
    frame_num: current frame number in observation
    background_sum: sum of current frames blue dimension across frames
    """
    
    if (frame_num > int(images.shape[0] / 2)
        and frame_num < (len(image_files) - int(images.shape[0] / 2))):

        replace_ind = image_ind + int(images.shape[0] / 2)
        replace_ind %= images.shape[0]
        
        # Subtract the pixel values that are about to be removed from background
        background_sum -= images[replace_ind, :, :, 2]
        
        image_file = image_files[frame_num + int(images.shape[0] / 2)]
        image = cv2.imread(image_file)
        images[replace_ind] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Add new pixel values to the background sum
        background_sum += images[replace_ind, :, :, 2]
            
    
    image_ind += 1
    # image_ind should always be in between 0 and images.shape - 1
    image_ind = image_ind % images.shape[0]
    
    return images, image_ind, background_sum

def initialize_image_array(image_files, focal_frame_ind, num_images):
    """ Create array of num_images x h x w x 3.
    
    Args:
        image_files (list): sorted paths to all image files in observation
        focal_frame_ind (int): number of the frame being process
        num_images (int): number of frames used for background subtraction
        
    return array, index in array where focal frame is located
    """
    
    images = []
    first_frame_ind = focal_frame_ind - (num_images // 2)
    if num_images % 2 == 0:
        # even
        last_frame_ind = focal_frame_ind + (num_images // 2) - 1
    else:
        # odd
        last_frame_ind = focal_frame_ind + (num_images // 2)
    for file in image_files[first_frame_ind:last_frame_ind+1]:
        image = cv2.imread(file)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    images = np.stack(images)
    focal_ind = num_images // 2
    
    
    return(images, focal_ind)
    
    


def process_frame(images, focal_frame_ind, bat_thresh, background_sum, bat_area_thresh, debug=False):
    """Process bat frame.
    
    images: n x h x w x c array where the n images are averaged together for background subtraction
    focal_frame_ind: which index in images array should be processed
    bat_thresh: float value to use for thresholding bat from background
    background_sum: sum of all blue channel pixels across the n dimension of images
    debug: if true return binary image

    """
    size_threshold = bat_area_thresh

    max_bats = 600
    mean = np.mean(images[focal_frame_ind, :, :, 2])
    if mean < 35:
        max_bats = 200
    if mean < 28:
        max_bats = 100

        
    if mean < 5:
        print('Too dark...')
        if debug:
            return None, None, None, None, None, None, None, None
        else:
            return None, None, None, None, None, None, None
    
    


    image_dif, background = subtract_background(images, focal_frame_ind, background_sum)
    
    
    while True:
        binary_image = preprocess_to_binary(image_dif, bat_thresh, background)

        bat_centers, bat_areas, contours, rect_angles, bat_sizes, bat_rects = get_blob_info(
            binary_image, background, size_threshold=size_threshold)
        
        if len(bat_centers) < max_bats:
            break
        bat_thresh += 0.05

    
    if debug:
        return bat_centers, bat_areas, contours, rect_angles, bat_sizes, bat_rects, bat_thresh, binary_image
    else:
        return bat_centers, bat_areas, contours, rect_angles, bat_sizes, bat_rects, bat_thresh
    
def add_all_points_as_new_tracks(raw_track_list, positions, contours, 
                                 sizes, current_frame_ind, noise):
    """ When there are no active tracks, add all new points to new tracks.
    
    Args:
        raw_track_list (list): list of tracks
        positions (numpy array): p x 2
        contours (list): p contours
        current_frame_ind (int): current frame index
        noise: how much noise to add to tracks initially
    """
    
    for ind, (position, contour, size) in enumerate(zip(positions, contours, sizes)):
        raw_track_list.append(
            ktf.create_new_track(first_frame=current_frame_ind, 
                                 first_position=position, pos_index=ind, 
                                 noise=noise, contour=contour, size=size 
            )
        )
        
    return raw_track_list
    
    

def find_tracks(first_frame_ind, positions, 
                contours_files=None, contours_list=None,
                sizes_list=None, max_frame=None, verbose=True, 
                tracks_file=None):
    """ Take in positions of all individuals in frames and find tracks.
    
    Args: 
        first_frame_ind (int): index of first frame of these tracks
        positions (list): n x 2 for each frame
        contours_files (list): list of files for contour info from each frame
        contours_list: already loaded list of contours, only used if contours_file
            is None
        sizes_list (list): sizes info from each frame
    
    return list of all tracks found
    """
    
    raw_track_list = []

    max_distance_threshold = 30
    max_distance_threshold_noise = 30
    min_distance_threshold = 0
    max_unseen_time = 2
    min_new_track_distance = 3
    min_distance_big = 30

#     #Create initial tracks based on the objects in the first frame
#     raw_track_list = add_all_points_as_new_tracks(
#         raw_track_list, positions[0], contours_list[0], sizes_list0, noise=0
#     )

    #try to connect points to the next frame
    if max_frame is None:
        max_frame = len(positions)
        
    contours_file_ind = 0
    previous_contours_seen = 0
    if contours_files:
        contours_list = np.load(contours_files[contours_file_ind], allow_pickle=True)
        while first_frame_ind >= previous_contours_seen + len(contours_list):
            contours_file_ind += 1
            previous_contours_seen += len(contours_list)
            contours_list = np.load(contours_files[contours_file_ind], allow_pickle=True)
        print(f'using {contours_files[contours_file_ind]}')   
    elif not contours_list:
        print("Needs contour_files or contour_list")
        return
        
    
    contours_ind = first_frame_ind - previous_contours_seen - 1
    
    
    for frame_ind in range(first_frame_ind, max_frame):
        contours_ind += 1
        
        if contours_files:
            if contours_ind >= len(contours_list):
                # load next file
                try:
                    contours_file_ind += 1
                    contours_list = np.load(contours_files[contours_file_ind], allow_pickle=True)
                    contours_ind = 0
                except:
                    if tracks_file:
                        tracks_file_error = os.path.splitext(tracks_file)[0] + f'-error-{frame_ind}.npy'
                        print(tracks_file_error)
                        np.save(tracks_file_error, np.array(raw_track_list, dtype=object))
        #get tracks that are still active (have been seen within the specified time)
        active_list = ktf.calculate_active_list(raw_track_list, max_unseen_time, frame_ind)
        
        if verbose:
            if frame_ind % 10000 == 0:
                print('frame {} processed.'.format(frame_ind))
                if tracks_file:
                    np.save(tracks_file, np.array(raw_track_list, dtype=object))
        if len(active_list) == 0:
            #No existing tracks to connect to
            #Every point in next frame must start a new track
            raw_track_list = add_all_points_as_new_tracks(
                raw_track_list, positions[frame_ind], contours_list[contours_ind], 
                sizes_list[frame_ind], frame_ind, noise=1
            )
            continue

        # Make sure there are new points to add
        new_positions = None
        row_ind = None
        col_ind = None
        new_sizes = None
        new_position_indexes = None
        distance = None
        contours = None
        if len(positions[frame_ind]) != 0:
            
            #positions from the next step
            new_positions = positions[frame_ind]
            contours = [np.copy(contour) for contour in contours_list[contours_ind]]
            new_sizes = sizes_list[frame_ind]
            
            raw_track_list = ktf.calculate_max_distance(
                raw_track_list, active_list, max_distance_threshold, 
                max_distance_threshold_noise, min_distance_threshold,
                use_size=True, min_distance_big=min_distance_big
            )

            distance = ktf.calculate_distances(
                new_positions, raw_track_list, active_list
            )
            
            max_distance = ktf.create_max_distance_array(
                distance, raw_track_list, active_list
            )
            
            assert distance.shape[1] == len(new_positions)
            assert distance.shape[1] == len(contours)
            assert distance.shape[1] == len(new_sizes)
                
            # Some new points could be too far away from every existing track
            raw_track_list, distance, new_positions, new_position_indexes, new_sizes, contours = ktf.process_points_without_tracks(
                distance, max_distance, raw_track_list, new_positions, contours, 
                frame_ind, new_sizes
            )
            
                
            if distance.shape[1] > 0:
                # There are new points can be assigned to existing tracks
                #connect the dots from one frame to the next
                
                row_ind, col_ind = linear_sum_assignment(np.log(distance + 1))
                
#                 for active_ind, track_ind in enumerate(active_list):
#                     if active_ind in row_ind:
#                         row_count = np.where(row_ind == active_ind)[0]
#                         raw_track_list[track_ind]['debug'].append(
#                             '{} dist {},  best {}'.format(
#                                 frame_ind,
#                                 distance[row_ind[row_count],
#                                          col_ind[row_count]],
#                                 np.min(distance[row_ind[row_count],
#                                          :])
#                             )
#                         )
#                         best_col = np.argmin(distance[row_ind[row_count],
#                                          :])
#                         row_count = np.where(col_ind == best_col)[0]
#                         raw_track_list[track_ind]['debug'].append(
#                             '{} row_ind {} col {} dist {} track {}'.format(
#                             frame_ind, row_ind[row_count],
#                             col_ind[row_count],
#                             distance[row_ind[row_count],
#                                          col_ind[row_count]],
#                             active_list[row_ind[row_count][0]])
#                         )
                        
                
                # In casese where there are fewer new points than existing tracks
                # some tracks won't get new point. Just assign them to 
                # the closest point
                row_ind, col_ind = ktf.filter_tracks_without_new_points(
                    raw_track_list, distance, row_ind, col_ind, active_list, frame_ind
                )
                # Check if tracks with big bats got assigned to small points which are
                # probably noise
                row_ind, col_ind = ktf.fix_tracks_with_small_points(
                    raw_track_list, distance, row_ind, col_ind, active_list, new_sizes, frame_ind)
                # see if points got assigned to tracks that are farther 
                # than max_threshold_distance
                # This happens when the closer track gets assigned 
                # to a differnt point
                row_ind, col_ind = ktf.filter_bad_assigns(raw_track_list, active_list, distance, max_distance,
                                                      row_ind, col_ind
                                                     )


        raw_track_list = ktf.update_tracks(raw_track_list, active_list, frame_ind, 
                                           row_ind, col_ind, new_positions, 
                                           new_position_indexes, new_sizes, contours, 
                                           distance, min_new_track_distance)
        raw_track_list = ktf.remove_noisy_tracks(raw_track_list)
    raw_track_list = ktf.finalize_tracks(raw_track_list) 
    if tracks_file:
        np.save(tracks_file, np.array(raw_track_list, dtype=object))
        print('{} final save.'.format(os.path.basename(os.path.dirname(tracks_file)))) 
    return raw_track_list
    
def get_tracked_bats_in_frame(image_files, focal_frame_ind, bat_thresh, bat_area_thresh):
    
    centers_list = []
    contours_list = []
    sizes_list = []
    
    clip_length = 5
    array_size = 31

    images, frame_buffer_ind = initialize_image_array(image_files, focal_frame_ind, array_size)

    background_sum = np.sum(images[:,:,:,2], 0, dtype=np.int16)

    for video_frame_ind in range(focal_frame_ind, focal_frame_ind+clip_length):
    
        bat_centers, bat_areas, bat_contours, _, _, _, bat_thresh = process_frame(
            images, frame_buffer_ind, bat_thresh, background_sum, 
            bat_area_thresh, debug=False)
        centers_list.append(bat_centers)
        contours_list.append(bat_contours)
        sizes_list.append(bat_areas)
        
        images, frame_buffer_ind, background_sum = update_circular_image_array(
            images, frame_buffer_ind, image_files, video_frame_ind, background_sum)
        
    raw_tracks = find_tracks(0, centers_list, 
                             contours_list=contours_list, 
                             sizes_list=sizes_list
                            )
    return raw_tracks, centers_list
    
#     return raw_tracks, centers_list, distance, max_distance, active_list, all_pre_distances, all_row_inds, all_col_inds

    
#     return(connected_distance, connected_size)

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], 
                        [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0]
                        )

def get_bat_accumulation(crossing_frames, obs=None, parameters=None, 
                         w_multiplier=True, w_darkness=True, w_frac=True):
    """ Create and return cummulative sum of bats crossing count line over the course of 
        list of given positive and negative crossing frames.
        
        crossing_frames: list of frame that each track crosses line. Positive if leaving
            negative if going
        obs: observation dictionary. 
        parameters: list of parameters of piecewise linear function
        w_multiplier: multiply each bat crossing by apropriate bat multiplier for camera etc.
        w_darkness: scale each bat crossing by apropriate accuracy corrrection based on frame darkness
        w_frac: scale each bat crossing by fraction of total circle that camera sees
    """
    if not np.any(crossing_frames):
        return np.zeros(1)
    last_crossing_frame = np.max(np.abs(crossing_frames))
    crossing_per_frame = np.zeros(last_crossing_frame+1)
    if obs and parameters:
        accurracies = piecewise_linear(obs['darkness'], *parameters)
        for crossing_frame, bm, acc in zip(crossing_frames, obs['multiplier'], accurracies):
            scale = 1
            if w_multiplier:
                scale *= bm 
            if w_darkness:
                scale *= (1/acc)
            if crossing_frame < 0:
                crossing_per_frame[-crossing_frame] -= scale
            elif crossing_frame > 0:
                crossing_per_frame[crossing_frame] += scale
        if w_frac:
            crossing_per_frame *= obs['fraction_total']
    else:
        for crossing_frame in crossing_frames:
            if crossing_frame < 0:
                crossing_per_frame[-crossing_frame] -= 1
            elif crossing_frame > 0:
                crossing_per_frame[crossing_frame] += 1
    return np.cumsum(crossing_per_frame)

def threshold_short_tracks(raw_track_list, min_length_threshold=2):
    """Only return tracks that are longer than min_length_threshold."""
    
    track_lengths = []
    track_list = []
    for track_num, track in enumerate(raw_track_list):
        if isinstance(track['track'], list):
            track['track'] = np.array(track['track'])
        track_length = track['track'].shape[0]
        if track_length >= min_length_threshold:
            track_lengths.append(track['track'].shape[0])
            track_list.append(track)
    return track_list

def calculate_height(wingspan_pixels, camera_constant, wingspan_meters):
    ''' Calculate bats height above the ground assumming wingspan_meters is correct.
    
    camera_constant = (frame pixels / 2) / tan(fov / 2) 
    
    height = constant * wingspan_meters / wingspan_pixels
    '''
    
    return camera_constant * wingspan_meters / wingspan_pixels

def calculate_bat_multiplier_simple(height, horizontal_fov, distance_to_center):
    ''' Calculate how many bats one bats at a given height and camera localtion represents.
    
        height: height of bat
        horizontal_fov: horizontal field of view of camera (degrees)
        distance_to_center: distance from camera to center of colony
        
        ASSUMES CIRCUMFERCE IS MUCH LARGER THAN WIDTH OF SPACE SEEN
        
        circumfernce c = 2 * pi * distance_to_center
        width of seen space w = 2 * height * tan(horizontal_fov / 2)
        
        multiplier = c / w
        '''
    
    c = 2 * np.pi * distance_to_center
    horizontal_fov_rad = horizontal_fov * np.pi / 180
    w = 2 * height * np.tan(horizontal_fov_rad / 2)
    return c / w

def calculate_bat_multiplier(height, horizontal_fov, distance_to_center):
    ''' Calculate how many bats one bats at a given height and camera 
    localtion represents.
    
        height: height of bat
        horizontal_fov: horizontal field of view of camera (degrees)
        distance_to_center: distance from camera to center of colony
        
        phi = arctan((height*tan(horizontal_fov/2)) / distance to center)
        
        multiplier = pi / phi
        '''
    
    horizontal_fov_rad = horizontal_fov * np.pi / 180
    distance_to_center = np.max([distance_to_center, 10e-5])
    phi = np.arctan((height * np.tan(horizontal_fov_rad / 2)) 
                    / distance_to_center
                   )
    return np.pi/phi

def combined_bat_multiplier(frame_width, wingspan_meters, 
                            wingspan_pixels, camera_distance):
    """ Calculates bat multiplier.
    
    Args:
        frame_width: frame width in pixels
        wingspan_meters: bat wingspan in meters
        wingspan_pixels: bat wingspan in pixels
        camera_distance: distance from forest point to camera in meters
            should be a single value or an array of distances with same
            shape as wingspan_pixels
        
    Returns:
        bat multiplier: float
    """
    denominator = np.arctan(
        (frame_width*wingspan_meters) 
        / (2*wingspan_pixels*camera_distance)
    )
    return np.pi / denominator

def get_rects(track):
    """ Fit rotated bounding rectangles to each contour in track.
    
    track: track dict with 'contour' key linked to list of cv2 contours
    """
    rects = []
    for contour in track['contour']:
        if len(contour.shape) >  1:
            rect = cv2.minAreaRect(contour)
            rects.append(rect[1])
        else:
            rects.append((np.nan, np.nan))
        
    return np.array(rects)

def get_wingspan(track):
    """ Estimate wingspan in pixels from average of peak sizes of longest
    rectangle edges.
    """
    
    if not 'rects' in track.keys():
        track['rects'] = get_rects(track)
                    
    max_edge = np.nanmax(track['rects'], 1)
    max_edge = max_edge[~np.isnan(max_edge)]
    peaks = signal.find_peaks(max_edge)[0]
    if len(peaks) != 0:
        mean_wing = np.nanmean(max_edge[peaks])
    else:
        mean_wing = np.nanmean(max_edge)
    
    return mean_wing

    
    

def measure_crossing_bats(track_list, frame_height=None, frame_width=None,
                          count_across=False, count_out=True, num_frames=None, 
                          with_rects=True):
    
    """ Find and quantify all tracks that cross middle line.
    
    track_list: list of track dicts
    frame_height: height of frame in pixels
    frame_width: width of frame in pixels
    count_across: count horizontal tracks
    count_out: count vertical tracks
    num_frames: number of frames in observation
    with_rects: if True calculate rects if not already
        in track and estimate wingspan and body size
    
    """
    if count_across:
        assert frame_width, "If vertical must specify frame width."
        across_line = CountLine(int(frame_width/2), line_dim=0, total_frames=num_frames)
    if count_out:
        assert frame_height, "If horizontal must specify frame height."
        out_line = CountLine(int(frame_height/2), line_dim=1, total_frames=num_frames)

    crossing_track_list = []

    for track_ind, track in enumerate(track_list[:]):
        out_result = None
        across_result = None
        if count_out:
            out_result, out_frame_num = out_line.is_crossing(track, track_ind)
        if count_across:
            across_result, across_frame_num = across_line.is_crossing(track, track_ind)
        if out_result or across_result:
            crossing_track_list.append(track)
            # result is 1 if forward crossing -1 is backward crossing
            if count_out:
                if out_frame_num:
                    crossing_track_list[-1]['crossed'] = out_frame_num * out_result
                else:
                    crossing_track_list[-1]['crossed'] = 0
            if count_across:
                if across_frame_num:
                    crossing_track_list[-1]['across_crossed'] = across_frame_num * across_result
                else:
                    crossing_track_list[-1]['across_crossed'] = 0
            track[id] = track_ind
            if with_rects:
                if not 'rects' in track.keys():
                    track['rects'] = get_rects(track)
                    
                min_edge = np.nanmin(track['rects'], 1)
                min_edge = min_edge[~np.isnan(min_edge)]
                peaks = signal.find_peaks(max_edge)[0]
                if len(peaks) != 0:
                    mean_body = np.nanmean(min_edge[peaks])


                else:
                    mean_body = np.nanmean(max_edge)

                crossing_track_list[-1]['mean_wing'] = get_wingspan(track)
                crossing_track_list[-1]['mean_body'] = mean_body

            
    return crossing_track_list

def get_camera_locations(observations, all_camera_locations, exclude=False):
    """Return dict of all camera locations that appear in observations.
    
    observations: dict of observations. Probably all observations from one day.
    all_camera_locations: dict containing all camera locations across all days
    exclude: if True, exclude observations as marked in obs dict
    """
    camera_locations = {}
    for camera, obs in observations.items():
        if exclude:
            if 'exclude' in obs.keys():
                if obs['exclude']:
                    continue
        camera_locations[obs['camera']] = all_camera_locations[obs['camera']]
    return camera_locations

def get_camera_distance(camera_utm, center_utm):
    """ Calculate the distance between utm of camera and possible
        forest center in meters.
    
    camera_utm: [x, y] array
    center_utm: [x, y] array
    """
    
    diff = camera_utm - center_utm
    return np.sum(np.sqrt(diff ** 2))
    

def get_camera_distances(camera_utms, center_utm):
    """ Calculate distance from every given camera to specified center.
    
    camera_utms: dict with camera names and locations
    center_utm: np.array 2d, location of forest center
    """
    
    camera_distances = {}
    for camera, camera_utm in camera_utms.items():
        camera_distances[camera] = get_camera_distance(camera_utm, 
                                                       center_utm)
    return camera_distances

def get_camera_angles(camera_utms, center_utm):
    """ Calculate angle from center point to each camera location.
    
    camera_utms: dict pairs of camera names and location info
    center_utm: 2d np.array, location of forest center
    """
    camera_angles = {}
    for camera, camera_utm in camera_utms.items():
        dif = camera_utm - center_utm
        camera_angles[camera] = np.arctan2(dif[1], dif[0])
    return camera_angles

def get_camera_borders(camera_utms, center_utm, jitter=False):
    """ Get angles around forest center that evenly bisect camera positions.
    
    camera_utms: dict pairs of camera names and location info
    center_utm: 2d np.array, location of forest center
    jitter: if True, don't actually bisect cameras at midpoint but drawn
        from a gaussian
    """
    
    camera_border = {}
    camera_angles = get_camera_angles(camera_utms, center_utm)
    for camera, camera_utm in camera_utms.items():
        min_neg = -10000
        min_pos = 100000
        # for border case where focal is positive angle 
        # and closest cclock is negative
        max_pos = 0 
        # for same case a last comment
        all_pos = True 
        # for border case where focal is positive angle 
        # and closest cclock is negative
        max_neg = 0 
        # for same case a last comment
        all_neg = True 
        max_camera = None
        camera_border[camera] = {'cclock': None,
                                 'cclock_angle': None,
                                 'clock': None,
                                 'clock_angle': None
                                }
        for alt_camera, alt_camera_utm in camera_utms.items():
            
            if camera == alt_camera:
                continue

            dif = camera_angles[camera] - camera_angles[alt_camera]
            if dif < 0:
                all_pos = False
                if dif > min_neg:
                    min_neg = dif
                    camera_border[camera]['cclock'] = alt_camera
                    camera_border[camera]['cclock_angle'] = dif / 2
                if dif < max_neg:
                    max_neg = dif 
                    max_camera = alt_camera

            if dif > 0:
                all_neg = False
                if dif < min_pos:
                    min_pos = dif
                    camera_border[camera]['clock'] = alt_camera
                    camera_border[camera]['clock_angle'] = dif / 2
                if dif > max_pos:
                    max_pos = dif 
                    max_camera = alt_camera

        if all_pos:
            camera_border[camera]['cclock'] = max_camera
            camera_border[camera]['cclock_angle'] = (max_pos - 2*np.pi) / 2
        if all_neg:
            camera_border[camera]['clock'] = max_camera
            camera_border[camera]['clock_angle'] = (max_neg + 2*np.pi) / 2
            
    if jitter:
        for camera, border_info in camera_border.items():
            camera_angle = camera_angles[camera]
            clockwise_camera = border_info['clock']
            angle_dif = border_info['clock_angle']
            # Three sttandard deviations is between camera pair
            jitter_angle = np.random.normal(scale=angle_dif/3)
            jitter_angle = np.maximum(-border_info['clock_angle'], 
                                      jitter_angle)
            jitter_angle = np.minimum(border_info['clock_angle'],
                                      jitter_angle)

            camera_border[camera]['clock_angle'] += jitter_angle
            if camera_border[camera]['clock_angle'] < 0:
                camera_border[camera]['clock_angle'] += (2 * np.pi)
            if camera_border[camera]['clock_angle'] >= (2 * np.pi):
                camera_border[camera]['clock_angle'] -= (2 * np.pi)
            camera_border[clockwise_camera]['cclock_angle'] += jitter_angle
            if camera_border[clockwise_camera]['cclock_angle'] < -2 * np.pi:
                camera_border[clockwise_camera]['cclock_angle'] += (2 * np.pi)
            if camera_border[clockwise_camera]['cclock_angle'] >= (2 * np.pi):
                camera_border[clockwise_camera]['cclock_angle'] -= (2 * np.pi)
        
            
    return camera_border

def latlong_dict_to_utm(latlong_dict):
    """ Convert dict of latlong coordinates to utm."""
    utm_dict = {}
    for key, latlong in latlong_dict.items():
        utm_val = utm.from_latlon(*latlong)
        utm_dict[key] = np.array([utm_val[0], utm_val[1]])
    return utm_dict

def get_camera_fractions(camera_utms, center_utm, jitter=False):
    """ Calculate the fraction of circle around center that each camera is closest to.
    
    camera_utms: dict of camera locations
    center_utm: 2d np array with utm coordinates of center
    jitter: If True instead of evenly dividing circle by
        cameras, set borders between camera from a gaussian
    
    return dict with fraction for each camera
    """
    if len(camera_utms) == 1:
        return {list(camera_utms.keys())[0]: 1.0}
    
    camera_borders = get_camera_borders(camera_utms, 
                                        center_utm,
                                        jitter=jitter)
    camera_fractions = {}
    for camera, border_info in camera_borders.items():
        angle = (-border_info['cclock_angle'] 
                 + border_info['clock_angle']
                )
        camera_fractions[camera] = angle / (np.pi * 2)
    return camera_fractions

def get_day_total(observations, center_utm, all_camera_utms, 
                  frame_width, wingspan, exclude=False, 
                  correct_darkness=False, parameters=None):
    """ Estimate total number of bats based on all observation counts
    and corespoinding camera locations.
    
    observations: dict of all observations for a specific day
    center_utm: estimated location of forest center
    all_camera_utms: dict of the utm locations of each camera
    frame_width: width of camera frame in pixels
    wingspan: estimated wingspan off all bats in meters
    exlude: to manually remove certain cameras, ie shut off early etc.
    correct_darkness: divide by accuracy estimated for given darkness
    parameters: param values of linear piecewise function for darkness
        error correction. Required if correct_darkness is True
    """
    
    frac_sum = 0
    total = 0
    obs_totals = []
    
    camera_utms = get_camera_locations(observations, all_camera_utms, exclude=True)
    camera_fractions = get_camera_fractions(camera_utms, center_utm)
    for obs in observations.values():
        if exclude:
            if 'exclude' in obs.keys():
                if obs['exclude']:
                    continue
        camera_distances = get_camera_distances(camera_utms, center_utm)
        obs['multiplier'] = combined_bat_multiplier(frame_width, 
                                                    wingspan, 
                                                    obs['mean_wing'], 
                                                    camera_distances[obs['camera']]
                                                   )
        if correct_darkness:
            assert parameters is not None, "Must pass parameters if correcting for darkness."
            acc = piecewise_linear(obs['darkness'], *parameters)
            obs['total_darkness'] = np.sum(obs['multiplier'] * obs['direction'] * (1/acc))
        obs['total'] = np.sum(obs['multiplier'] * obs['direction'])
        obs['total_unscaled'] = np.sum(obs['direction'])
        obs['fraction_total'] = camera_fractions[obs['camera']]
        frac_sum += obs['fraction_total']
        if correct_darkness:
            total += obs['total_darkness'] * obs['fraction_total']
            obs_totals.append(obs['total_darkness'])
        else:
            total += obs['total'] * obs['fraction_total']
            obs_totals.append(obs['total'])

    if len(obs_totals) > 0:
        mean_total = np.mean(obs_totals)
    else:
        mean_total = 0

    return total, mean_total

def get_peak_freq(raw_freqs, raw_powers, min_freq):
    """ Calculate max power frequency above min_freq.
    
    raw_freqs: list of frequencies
    raw_powers: list of powers assosiated with each raw freq value
    min_freq: minimum acceptable frequency value
    """
    
    freqs = raw_freqs[raw_freqs>min_freq]
    powers = raw_powers[raw_freqs>min_freq]
    
    if np.any(np.isnan(freqs)) or len(freqs)==0:
        return np.nan, np.nan
    
    return freqs[np.argmax(powers)], powers[np.argmax(powers)]



def get_track_wingbeat_freqs(track, fps=25, min_freq=.75):
    """ Calculate peak wing freqs and assosiated power.
    
    track: track dict
    fps: frames per second track temporal resolution
    min_freq: minimum frequency for calculating peak_freq.
        Messily segmented tracks often have have high power
        close to 0 Hz because actual signal is not clear.
    """
    
    assert 'max_edge' in track.keys(), "Track must have max_edge already computed"
    
    if len(track['max_edge']) < 255:
        nperseg = len(track['max_edge'])
    else:
        nperseg = 255

    f, p = signal.welch(track['max_edge'], fps, nperseg=nperseg)
    peaks = signal.find_peaks(p, threshold=0, height=1)[0]

    track['freqs'] = f[peaks]
    track['freqs_power'] = p[peaks]
    peak_freq, freq_power  = get_peak_freq(track['freqs'],
                                       track['freqs_power'],
                                       min_freq
                                      )
    track['peak_freq'] = peak_freq
    track['peak_freq_power'] = freq_power
    
def add_wingbeat_info_to_tracks(tracks, fps=25, min_freq=.75, 
                                remove_contours=False):
    """ Add main wingbeat freq info for all tracks in tracks after calculating
    all nessissary extra info. Can remove contours after getting bounding rects 
    to save memory.
    
    tracks: list of track dicts
    fps: frames per second - temporal resolution of tracks
    min_freq: minimum frequency for calculating peak_freq.
        Messily segmented tracks often have have high power
        close to 0 Hz because actual signal is not clear.
    remove_contours: if True remove raw contour info from track dicts.
        Useful if need to save memory
    """
    for track in tracks:
        if 'rects' not in track.keys():
            track['rects'] = get_rects(track)
        if remove_contours:
            try:
                del track['contour']
            except KeyError:
                pass
                
        if 'max_edge' not in track.keys():
            track['max_edge'] = np.nanmax(track['rects'], 1)
        if 'mean_wing' not in track.keys():
            track['mean_wing'] = get_wingspan(track)
        
        get_track_wingbeat_freqs(track, fps=fps, min_freq=min_freq)
        
def get_random_utm_in_mask(mask, rasterio_map, num_locations=1):
    """ Get a random utm location within raster mask.
    
    mask: 2d np array where forest has values > 0 and background < 0
    rasterio_map: rasterio.io.DatasetReader for mask
    num_locations: number of locations to return in forest
    """
    
    in_hull = np.argwhere(mask>0)
    ind = np.random.randint(0, in_hull.shape[0], num_locations)
    area_x_origin = rasterio_map.bounds.left
    area_y_origin = rasterio_map.bounds.bottom
    
    xutm = in_hull[ind, 1] + area_x_origin
    yutm = in_hull[ind, 0] + area_y_origin
    
    utm_vals = np.stack([xutm, yutm], axis=1)
    
    # squeeze for when only returning one value to remove
    # extra dimension
    return np.squeeze(utm_vals)
        
        
def get_wing_correction_distributions(validation_file, num_darkness_bins, 
                                      kde_bw_scale=1, should_plot=False):
    """ Calculate wing correction distributions from human validation info.
    
    validation_file: .csv file with human groundtruth info
    num_darkness_bins: how many groups to split darkness range into
    kde_bw_scale: kernel size used in kde calculation: data std. in bin * kde_bw_scale
    should_plot: show histograms and resulting distributions
    """
    wing_validation = pd.read_csv(validation_file)
    max_darkness = wing_validation.loc[wing_validation['has_gt'], 'darkness'].max()

    darkness_bins = np.linspace(0, max_darkness, num_darkness_bins+1)
    darkness_bins[-1] = 255
    wing_correction_kdes = []

    for bin_num in range(num_darkness_bins):
        rows_in_bin = (wing_validation['has_gt'] 
                       & (wing_validation['darkness'] > darkness_bins[bin_num])
                       & (wing_validation['darkness'] <= darkness_bins[bin_num+1])
                       & (wing_validation['error_norm'] > -1)
                      )
        errors = wing_validation.loc[rows_in_bin, 'error_norm'].values
        error_std = errors.std()

        kde = KernelDensity(
            kernel='gaussian', bandwidth=error_std*kde_bw_scale).fit(errors[..., np.newaxis])
        wing_correction_kdes.append(kde)
        if should_plot:

            sorted_error = np.sort(errors, axis=0)
            samples = np.linspace(-1,1,100)
            log_dens = kde.score_samples(samples[..., np.newaxis])

            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.hist(sorted_error, bins=40, density=True)
            ax1.plot(samples, np.exp(log_dens), c='cyan')
    if should_plot:
        plt.figure()
        for kde in wing_correction_kdes:
            samples = np.linspace(-1,1,100)
            log_dens = kde.score_samples(samples[..., np.newaxis])
            plt.plot(samples, np.exp(log_dens),)
    
    return wing_correction_kdes, darkness_bins

def get_kde_samples(obs, kde_list, darkness_bins):
    """ Draw a sample for each track from appropriate kde distribution for tracks darkness.
    
    obs: observation dictionary
    kde_list: a kde distribution for each darkness bin
    darkness_bins: list of darkness thresholds for between each darkness bin
        starts at zero so len=num bins + 1
    """
    kde_samples = np.zeros(len(obs['darkness']))
    kde_inds = np.zeros(len(obs['darkness']))
    for ind, (kde, min_bin_val, max_bin_val) in enumerate(zip(kde_list, darkness_bins[:-1], darkness_bins[1:])):
        inds_in_bin = ((obs['darkness'] > min_bin_val) 
                       & (obs['darkness'] <= max_bin_val))
        bin_samples = np.squeeze(kde.sample(len(obs['darkness'])))
        kde_samples[inds_in_bin] = bin_samples[inds_in_bin]
        kde_inds[inds_in_bin] = ind
        
    return kde_samples, kde_inds
        

def correct_wingspan(estimate, estimate_scale):
    """ Correct the estimated wingspan based on groundtruth distribution.
    
    
    estimate: wingespan estimated from track
    estimate_scale: (estimate - groundtruth)/ estimate
        obv. don't know groundtruth in application but 
        estimate scale usually ranomly drawn from distribution
    """
    
    corrected_est = estimate - estimate * estimate_scale
    
    return corrected_est

def save_fig(save_folder, plot_title, fig=None):
    """ Convient default figure saving configuration."""
    plot_name = plot_title.replace(' ', '-')
    file = os.path.join(save_folder, plot_name+'.png')
    if fig:
        fig.savefig(file, bbox_inches='tight', dpi=600)
        return
    
    plt.savefig(file, bbox_inches='tight', dpi=600)
    
def smooth_track(track, kernel_size=12):
    """ Smooth n x 2 track with averaging filter."""
    
    kernel = np.ones(kernel_size) / kernel_size

    x = np.convolve(track[:, 0], kernel, mode='valid')
    y = np.convolve(track[:, 1], kernel, mode='valid')
    
    return np.stack([x, y], 1)
    

def calculate_straightness(track):
    """ Caclute straightness of n x 2 numpy track."""
    
    track = smooth_track(track, kernel_size=12)
    
    step_vectors = track[1:] - track[:-1]
    step_sizes = np.linalg.norm(step_vectors, axis=1)
    combined_steps = np.sum(step_sizes)
    net_distance = np.linalg.norm(track[-1] - track[0])
    
    return net_distance / combined_steps

def get_middle_percentiles(values, lower_percentile, upper_percentile):
    """ Return all values in values between lower and upper percentile."""
    values = np.array(values)
    values = values[~np.isnan(values)]
    sorted_values = sorted(values)
    lower_ind = int(lower_percentile * len(values))
    upper_ind = int(upper_percentile * len(values) + 1)
    
    return sorted_values[lower_ind:upper_ind]

def calc_movement_unit_vector(track, frame_height=1520):
    """ Calculate the unit vector pointing from first position
        to last position in track with bottom left origin
    
    track: track dict
    frame_height: height of frame in pixels the tracks came from
    """
    track = np.copy(track['track'])
    track[:, 1] = frame_height - track[:, 1]
    diff = track[-1] - track[0]

    unit_position = diff / np.linalg.norm(diff)
    
    return unit_position

def calculate_polarization(tracks):
    """ Following Couzin et al. 2002 calculate polarization of all bats
        in tracks.
    """
    direction_unit_vectors = []

    for track in tracks:
        direction_unit_vectors.append(
            calc_movement_unit_vector(track))
    direction_sum = np.sum(np.array(direction_unit_vectors), axis=0)
    direction_magnitude = np.linalg.norm(direction_sum) 
    polarization = direction_magnitude / len(tracks)
    
    return polarization

def get_camera_color_dict(colormap=plt.cm.tab10):
    """ For consistent camerar colors across plots."""
    camera_colors = {'FibweParking2': colormap(0),	
                    'FibweParking': colormap(0),
                    'Chyniangale': colormap(.1),	
                    'BBC': colormap(.2),
                    'Sunset': colormap(.3),
                    'NotChyniangale': colormap(.4),
                    'MusoleParking': colormap(.5),	
                    'MusolePath2': colormap(.6),	
                    'MusolePath': colormap(.6),
                    'Puku': colormap(.7),	
                    'FibwePublic': colormap(.8),	
                    'MusoleTower': colormap(.9),
                    }
    return camera_colors
    