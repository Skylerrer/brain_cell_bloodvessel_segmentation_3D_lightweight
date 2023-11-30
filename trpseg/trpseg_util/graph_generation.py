"""
Code for Masterthesis "Segmenting Blood Vessels and TRP Channel-Expressing Cells in 3D Light Sheet Fluorescence Microscopy Images of Mouse Brains"
Author: Mischa Breit
Year: 2023
E-Mail: mbreit11@web.de
"""

########################################################################################################################
# Most of the code here is just modified code from:
# 1. Christoph Kirst: Paper "Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature"
#                     Code https://github.com/ChristophKirst/ClearMap2/
# 2. Jacob Bumgarner: Paper "Open-source analysis and visualization of segmented vasculature datasets with VesselVio"
#                     Code https://github.com/JacobBumgarner/VesselVio/
# Exact references are given above each method

# Our work consists of modifying their code for our needs.
# The methods: skeletonize3D, compute_radii_dmap, prune_short_end_branches, perform_graph_generation, reduced_graph_stats_summary
# and further methods that have no reference above them are also our work
########################################################################################################################

######################################################File Content######################################################
# This file implements the algorithms to generate a graph from a given binary filled blood vessels segmentation.
# First 3D skeletonization is performed.
# Afterwards a graph is generated from the skeleton. The graph stores features for blood vessel branches.
# Currently not all features are written to a txt but only the most robust ones that we are currently interested in: vasculature length, vessel radii, vessel volume
########################################################################################################################

import os
import time
from threading import Event
from collections import namedtuple

import numpy as np
import igraph as ig
from pathlib import Path
from timeit import default_timer as timer
from skimage.morphology import skeletonize_3d

from trpseg import OUTPUT_DIRECTORY
from trpseg.trpseg_util.utility import read_image, save_image, store_substack_excluding_radius, get_file_list_from_directory, get_z_splits, read_img_list


#Skeletonization
def skeletonize3D(input_folder_path, output_folder_path, resolution, max_images_in_memory=200, isTif=False, channel_one=None, canceled=Event()):
    """Perform 3D skeletonization on a binary image stack.

    This algorithm performs skeletonization in blocks of max_images_in_memory
    and uses overlap to prevent boundary artifacts.

    Parameters
    ----------
    input_folder_path: str, pathlib.Path
        The path to the folder containing the binary image slices to be processed.
    output_folder_path: str, pathlib.Path
        The path to the folder where the output images are stored
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    max_images_in_memory : int
        The maximum number of input images loaded into the memory and processed at once
    isTif : bool, optional
        Determines, which input images inside input_folder_path are considered.
        True, means that images have file type .tif
        False, means that images have file type .png
        None, means that all file types are considered
    channel_one : bool, optional
        Determines, which channel from the input images to use.
        True, means that only images containing the string "_C01_" are considered.
        False, means that only images containing the string "_C00_" are considered.
        None, means that all channels are considered.
    canceled : threading.Event, optional
        An event object that allows to cancel the process

    Notes
    ----------
    This method uses the skeletonize_3d method from skimage which implements the algorithm from the following paper:
    "Building Skeleton Models via 3-D Medial Surface Axis Thinning Algorithms", Lee et. al. 1994
    https://www.sciencedirect.com/science/article/abs/pii/S104996528471042X
    """
    input_files_list = get_file_list_from_directory(input_folder_path, isTif=isTif, channel_one=channel_one)

    # input_files_list = input_files_list[268:269]

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    number_images = len(input_files_list)

    # An overlap of 20 worked well for images with resolution in z direction larger or equal to 1 micrometer.
    # Experiments on “unet_res_11_46_06_kaggle200cplx_c2_post_smooth” with 200 images overlap 20 returned exactly
    # same result as when computing skeletonize_3d on complete stack at once.

    overlap = get_overlap_for_skeletonization(resolution)

    splits = get_z_splits(number_images, max_images_in_memory, overlap)

    for start, end in splits:

        if canceled.is_set():
            return

        process_files = input_files_list[start:end]

        imgStack = read_img_list(process_files, dtype=np.uint8)
        print(f"Start skeletonize stackpart [{start},{end}]")
        skel = skeletonize_3d(imgStack)

        store_substack_excluding_radius(skel, start, end, overlap, number_images, process_files, output_folder,
                                        dtype=np.uint8, file_extension=".png")


def get_overlap_for_skeletonization(resolution):
    """ Return the number of z-slices used as overlap when performing block-wise skeletonization.

    Parameters
    ----------
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    """

    z_res = resolution[0]

    overlap = int(20/z_res)

    return overlap


# Idea from:
######################################################################################
#   Title: Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature
#   Authors: Christoph Kirst
#   Date: 20th February 2020
#   Availability: https://www.sciencedirect.com/science/article/pii/S0092867420301094
#   Similar Code: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/ImageProcessing/Topology/Topology3d.py in method orientations(...)
######################################################################################

# Code modified from:
######################################################################################
#   Title: orientations
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/graph_processing.py
######################################################################################
def orientations13():
    """Get the relative indices of selected 13 neighbors around a center voxel in a 3x3x3 cube

    Returns
    ----------
    scan : np.ndarray of shape (13, 3)
        The thirteen selected relative neighbors.

    Notes
    ----------
    This method is used to generate the graph edges in the skeletonized binary image stack.
    The idea is based on C. Kirst's algorithm. See reference above.

    The point of interest rests at [1,1,1] in 3x3x3 cube.
    We have the convention to index the array according to the following order [z, y, x].
    z is image slice index and y, x are the coordinates in one image slice (y axis vertical, x axis horizontal).

    By only looking at 13 directions of the 26-neighborhood no double edges are generated.
    Each pair of voxels is only considered once when one iterates over the image stack.
    """

    scan = np.array(
        [
            [0, 1, 1], #U
            [1, 1, 0], #W
            [1, 2, 1], #S
            [1, 2, 0], #SW
            [1, 2, 2], #ES
            [0, 0, 1], #UN
            [0, 1, 0], #UW
            [0, 1, 2], #UE
            [0, 2, 1], #US
            [0, 0, 0], #UNW
            [0, 0, 2], #UNE
            [0, 2, 0], #USW
            [0, 2, 2], #USE
        ]
    )

    scan -= 1 #Important to get relative indices around center voxel of cube

    return scan


def create_vertices_dict(points):
    """Create a dictionary from the points list to allow for fast lookup of a points index.

        Parameters
        ----------
        points : (N, 3) list
            List of 3 dimensional point coordinates

        Returns
        ----------
        A dictionary that maps 3D coordinates to their index in the points list.
        """

    #python dictionary uses hash mapping and one can therefore efficiently query elements
    v_dict = {}

    i = 0
    for point in points:
        key = str(point[0]) + "_" + str(point[1]) + "_" + str(point[2])
        v_dict[key] = i
        i += 1

    return v_dict


# compute radii given a already computed distance map
# distance map can be a memory map
def compute_radii_dmap(points, edtDistanceMap):
    """Get the distance at the given points from a given distance map.

    This method is used to get the radii of blood vessels where points are the center skeleton line points of blood vessels.

    Parameters
    ----------
    points : (N, 3) list
        List of 3 dimensional point coordinates
    edtDistanceMap : np.ndarray
        3-dimensional image stack that specifies at each voxel how far it is from background (non-vessel voxel)

    Returns
    -------
    A list containing the distance of point i at index i.

    Notes
    -------
    Since distance map often store distances as float32 or float64 they can consume a lot of memory.
    The edtDistanceMap can therefore also be given to this method as a numpy memory map.
    """

    radii = np.empty(shape=(len(points),), dtype=np.float32)

    for i in range(len(points)):
        p = points[i]
        radii[i] = edtDistanceMap[p[0], p[1], p[2]]

    return radii


# Modified from:
######################################################################################
#   Title: measure_radius
#   Authors: Christoph Kirst
#   Date: 12th May 2023
#   Availability: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/Analysis/Measurements/MeasureRadius.py
######################################################################################
def compute_radii_points(binary_volume, points, max_radius, scale):
    """For every point in the given points list compute its distance to the background (0 valued voxels) in the binary_volume.

    Parameters
    ----------
    binary_volume : np.ndarray
        The binary 3d volume representing the complete blood vessel segmentation result.
    points : list of shape (N, 3)
        List of 3 dimensional point coordinates
    max_radius : int
        The maximal radius in micrometers of a ball in which to search for a background voxel around points.
    scale : (3,) array, tuple
        A tuple or array specifying the resolution in order [z, y, x]

    Returns
    -------
    radii: (N, 3) list
        A list containing the blood vessel radii at points from the point list.
        At index i the radius from location of point i can be found.

    Notes
    -------
    binary_volume can be given as numpy memory map if you do not have enough memory available.
    """

    pixel_radius_z = int(np.ceil(max_radius/scale[0]))
    pixel_radius_xy = int(np.ceil(max_radius/scale[1]))
    max_radius = (pixel_radius_z, pixel_radius_xy, pixel_radius_xy)

    sorted_search_idx = search_indices_sphere(max_radius)

    start = timer()
    radii = distance_closest_background_pixel(binary_volume, points, sorted_search_idx, scale)
    end = timer()
    print(f"Radii computation took: {end-start} seconds")
    return radii




# Modified from:
######################################################################################
#   Title: find_smaller_than_value
#   Authors: Christoph Kirst
#   Date: 12th May 2023
#   Availability: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/ParallelProcessing/DataProcessing/MeasurePointList.py
######################################################################################
def distance_closest_background_pixel(binary_volume, points, sorted_search_indices, scale):
    """For every point in the given points list compute its distance to the background (0 valued voxels) in the binary_volume.

    The neighborhood of each point is searched for background voxels at the given relative search indices in sorted_search_indices.

    Parameters
    ----------
    binary_volume : np.ndarray
        The binary 3d volume representing the complete blood vessel segmentation result.
    points : list of shape (N, 3)
        List of 3 dimensional point coordinates
    sorted_search_indices : list of shape (M, N, 3)
        List of relative coordinates sorted after their euclidean distance
    scale : (3,) array, tuple
        A tuple or array specifying the resolution in order [z, y, x]

    Returns
    -------
    radii: (N, 3) list
        A list containing the blood vessel radii at points from the point list.
        At index i the radius from location of point i can be found.
    """

    numPoints = points.shape[0]
    scale = np.asarray(scale, dtype=np.float32)

    radii = np.zeros(shape=(numPoints,), dtype=np.float32)

    stack_shape = binary_volume.shape

    for i in range(0, numPoints):

        if not i % 1000:
            print(f'Compute Radii {i:d}/{numPoints:d}')

        p = points[i]
        radii[i] = -1

        found_radii = False
        for s_idx in sorted_search_indices:

            check_p = p + s_idx
            if check_p[0] >= stack_shape[0] or check_p[0] < 0:
                continue
            if check_p[1] >= stack_shape[1] or check_p[1] < 0:
                continue
            if check_p[2] >= stack_shape[2] or check_p[2] < 0:
                continue

            if binary_volume[check_p[0], check_p[1], check_p[2]] == 0:
                radius = check_p-p
                radius = (radius * scale)
                radius = np.sqrt(np.sum(radius * radius))
                radii[i] = radius
                found_radius = True
                break

        if not found_radius:
            raise RuntimeError("The max_radius parameter is choosen too small! Choose a larger value!")

    return radii


# Code modified from:
######################################################################################
#   Title: search_indices_sphere
#   Authors: Christoph Kirst
#   Date: 12th May 2023
#   Availability: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/Analysis/Measurements/MeasureRadius.py
######################################################################################
def search_indices_sphere(radius):
    """Creates all relative indices within a sphere of specified radius.

    Parameters
    ---------
    radius : tuple of int of shape (3,)
      A tuple of three integers, which specify the max radius for which to create search indices.
      Order is [z, y, x]

    Returns
    -------
    indices : np.ndarray
       List of relative coordinates around the middle of a circle sorted after their euclidean distance
    """

    #Generate as many lists as radius has dimensions (lists have normalized distance in each direction)
    #Handles anisotropic resolution
    #Sign is unimportant here because we square the numbers anyways
    grid = [np.arange(-r, r + 1, dtype=float) / np.maximum(1, r) for r in radius]

    # Arrange values in grid
    # Axis 0 are the three dimensions
    # Assume shape (3,5,7,7)
    # ([0,0,0,0], [1,0,0,0], [2,0,0,0]) is basically the vector that points to point [0,0,0,0] in the grid (this example vector would have value (1,1,1))
    # Attention!: the middle point is in ([0,2,3,3], [1,2,3,3], [2,2,3,3]) (this example vector would have value (0,0,0))
    grid = np.array(np.meshgrid(*grid, indexing='ij'))

    # sort indices by radius
    # to get the distance to the middle -> square the grid and sum up -> squared euclidean distance
    dist = np.sum(grid * grid, axis=0)
    dist_shape = dist.shape
    dist = dist.reshape(-1)
    # sort after squared distance around middle point
    dist_index = np.argsort(dist)
    dist_sorted = dist[dist_index]
    # only keep points in circle -> euclidean distance <= 1 (1 and not parameter radius because we normalized the values before)
    keep = dist_sorted <= 1
    dist_index = dist_index[keep]

    # convert to relative coordinates
    indices = np.array(np.unravel_index(dist_index, dist_shape)).T
    indices -= radius
    indices = indices.astype(np.int32)

    #return the relative coordinates around the middle of a circle sorted after their euclidean distance
    return indices


# Code modified from:
######################################################################################
#   Title: identify_edges
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/graph_processing.py
######################################################################################
def generate_edges(points, vertex_LUT, orientations):
    """Generate pairs of points from the points list, that are 26-connected to each other.

    Parameters
    ----------
    points : list of shape (N, 3)
        List of 3 dimensional point coordinates
    vertex_LUT : dict
        A dictionary that maps 3D coordinates to their index in the points list == their vertex id
    orientations : np.ndarray
        The 13 relative neighbors to search for points that are connected to the current one.
    Returns
    -------
    edges : list of shape (M, 2)
        The connections between neighboring points.
    """
    edges = []
    for i in range(points.shape[0]):
        neighbor13idx = orientations + points[i]

        for j in range(13):
            key = str(neighbor13idx[j][0]) + "_" + str(neighbor13idx[j][1]) + "_" + str(neighbor13idx[j][2])
            neighborID = vertex_LUT.get(key)
            if neighborID != None:
                edges.append((i, neighborID))
    return edges


def read_pickled_graph(filename):
    """
    Read a pickled igraph from the specified file.
    """
    g = ig.Graph.Read_Pickle(filename)

    return g


# Code modified from:
######################################################################################
#   Title: create_graph
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/graph_processing.py
######################################################################################
def generate_skeleton_graph(skeleton_path, binary_path, resolution, max_radius=None, edtDistanceMap=None, work_memory_efficient=False, verbose=False):
    """
    Generate a graph from a blood vessels skeleton. Skeleton points are vertices and edges are pairs of 26-connected skeleton points.

    If work_memory_efficient is set to True and binary_path is a path to a .npy file then this algorithm uses much less memory.
    However, it will then also be much slower.

    Parameters
    ----------
    skeleton_path: str, pathlib.Path
        Path to a .npy file or folder containing the skeleton images.
    binary_path : str, pathlib.Path
        Path to a .npy file or folder containing the binary blood vessel segmenation images.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    max_radius : int, optional
        The maximal radius in micrometers of a ball in which to search for a background voxel around points.
        If None, default value of 30 micrometers is used
    edtDistanceMap : np.ndarray, np.memmap, optional
        3-dimensional image stack that specifies at each voxel how far it is from background (non-vessel voxel) in the filled blood vessel binary.
        This can be used to compute the radius of the vessels which will be extracted here for every skeleton point.
        Can also be a numpy memmap. This makes it possible to use distance maps that do not fit into memory at once.
        If None, then radii are computed by searching neighborhood of every skeleton point for closest background voxel
    work_memory_efficient : bool
        If True, then this method works more memory efficient. It asserts that binary_path is a npy file.
    verbose : bool
            Determines whether to print status messages.

    Returns
    -------
    g : ig.Graph
        Graph generated from skeleton. Skeleton points are vertices and edges are pairs of 26-connected skeleton points.
    """

    skeleton_path = Path(skeleton_path)
    binary_path = Path(binary_path)

    if binary_path.parts[-1][-3:] != "npy" and work_memory_efficient is True:
        raise RuntimeError("If work_memory_efficient is set to True, the binary path must point to a .npy file.")

    if max_radius is None:
        max_radius = 30 #in experiments a max_radius of 30 micrometers was enough

    skel_volume_is_from_npy = False
    if len(skeleton_path.name) > 3: #just to make sure that no indexing error occurs in next step
        if skeleton_path.parts[-1][-3:] == "npy":
            if work_memory_efficient:
                skeleton_volume = np.load(skeleton_path, mmap_mode='r')
            else:
                skeleton_volume = np.load(skeleton_path)
            skel_volume_is_from_npy = True

    if not skel_volume_is_from_npy and not work_memory_efficient:
        skel_files_list = get_file_list_from_directory(skeleton_path)
        skeleton_volume = read_img_list(skel_files_list, dtype=np.uint8)

        points = np.where(skeleton_volume)
        points = np.stack(points, axis=-1)
        points = points.astype(np.uint16)

    elif not skel_volume_is_from_npy and work_memory_efficient:
        skel_files_list = get_file_list_from_directory(skeleton_path)

        points = np.empty(shape=(0, 3), dtype=np.uint16)

        for i in range(len(skel_files_list)):
            img = read_image(skel_files_list[i])

            current_points = np.argwhere(img)
            current_points = current_points.astype(np.uint16)

            #prepend current z coordinate
            current_points_num = len(current_points)
            if current_points_num > 0:
                prepend_this = np.full(shape=(len(current_points), 1), fill_value=i, dtype=np.uint16)
                current_points = np.concatenate((prepend_this, current_points), axis=1)

                points = np.concatenate((points, current_points), axis=0)


    skeleton_volume = None #allow earlier garbage collection

    bin_volume_loaded = False
    if len(binary_path.name) > 3: # just to be sure no indexing error occurs in next step
        if binary_path.parts[-1][-3:] == "npy":
            if work_memory_efficient:
                binary_volume = np.load(binary_path, mmap_mode='r')
            else:
                binary_volume = np.load(binary_path)
            bin_volume_loaded = True

    if not bin_volume_loaded:
        binary_files_list = get_file_list_from_directory(binary_path)
        binary_volume = read_img_list(binary_files_list, dtype=np.uint8)

    orientations = orientations13()

    vertices_LUT = create_vertices_dict(points)

    edges = generate_edges(points, vertices_LUT, orientations)

    if edtDistanceMap is not None:
        radii = compute_radii_dmap(points, edtDistanceMap)
    else:
        radii = compute_radii_points(binary_volume, points, max_radius, resolution)

    points = points.astype(np.float32)

    g = ig.Graph(n=len(points), edges=edges, directed=False,
                 vertex_attrs={'coord': points, 'radius': radii})

    return g


#Removes cliques of branch points (points with more than two neighbors)
#Removes duplicate edges and self-loops
#Removes isolated vertices

# Code modified from:
######################################################################################
#   Title: clean_graph
#   Authors: Christoph Kirst
#   Date: 12th May 2023
#   Availability: https://github.com/ChristophKirst/ClearMap2/blob/master/ClearMap/Analysis/Graphs/GraphProcessing.py
######################################################################################
def clean_graph(g: ig.Graph, remove_isolated_vertices=True):
    """Clean the graph: Resolves cliques of branching points. Removes self-loops, double edges and possibly isolated vertices.

    Parameters
    ----------
    g : ig.Graph
        The igraph Graph to be cleaned.
    remove_isolated_vertices : bool, optional
        If True, remove isolated vertices

    Returns
    -------
    cliqueless_g : ig.Graph
        The cleaned graph.
    """


    num_vertices = g.vcount()
    branch_points = g.vs.select(_degree_gt=2)
    branch_graph = g.induced_subgraph(branch_points)

    branch_point_clusters = ig.Graph.connected_components(branch_graph) #returns igraph VertexClustering

    cluster_ids = []
    for i in range(len(branch_point_clusters)):
        if len(branch_point_clusters[i]) > 1:
            cluster_ids.append(i)

    num_clusters = len(cluster_ids)

    cliqueless_g = g.copy()
    cliqueless_g.add_vertices(num_clusters)

    vertex_seq = cliqueless_g.vs

    to_remove = []
    for i in range(num_clusters):
        #print(i)
        vertex_id = i + num_vertices
        ci = cluster_ids[i]
        c_members = branch_point_clusters[ci]

        #get corresponding ids in cliqueless_g (before, ids were from branch_graph)
        c_members_orig_ids = [branch_points[i].index for i in c_members]

        to_remove.extend(c_members_orig_ids)


        neighbors = np.hstack([cliqueless_g.neighbors(m) for m in c_members_orig_ids])
        neighbors = np.setdiff1d(np.unique(neighbors), c_members_orig_ids)

        # connect to new node
        cliqueless_g.add_edges([[n, vertex_id] for n in neighbors])


        coords = []
        r = []
        for m in c_members_orig_ids:
            coords.append(vertex_seq[m]["coord"])
            r.append(vertex_seq[m]["radius"])

        # set coordinate as the mean coordinate of the clique members
        vertex_seq[vertex_id]["coord"] = np.mean(coords, axis=0)

        #set radius as the maximum radius of the clique members
        vertex_seq[vertex_id]["radius"] = np.max(r)


    #delete old clique vertices
    remove_ids = np.asarray(to_remove)
    remove_ids = np.unique(remove_ids)
    cliqueless_g.delete_vertices(remove_ids)

    print('Graph cleaning: removed %d cliques of branch points from %d to %d nodes and %d to %d edges' % (num_clusters, len(g.vs), len(cliqueless_g.vs), len(g.es), len(cliqueless_g.es)))

    # Remove multiple edges and self-loops (probably not necessary but better to be safe than sorry)
    cliqueless_g.simplify(multiple=True, loops=True)

    print('Graph simplify: Now there are %d nodes and %d edges' % (len(cliqueless_g.vs), len(cliqueless_g.es)))

    #Remove isolated vertices that have no edges to other vertices
    if remove_isolated_vertices:
        non_isolated = cliqueless_g.vs.select(_degree_gt=0)
        num_before = len(cliqueless_g.vs)
        cliqueless_g = cliqueless_g.induced_subgraph(non_isolated)
        print(f'Removed {num_before - len(non_isolated)} isolated vertices.')


    return cliqueless_g


# Code modified from:
######################################################################################
#   Title: vgraph_analysis
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/feature_extraction.py
######################################################################################
def reduce_graph(g, resolution):
    """
    Reduce the graph generated from a skeleton.

    Parameters
    ----------
    g : ig.Graph
        Graph that should be reduced.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Returns
    -------
    The reduced graph of type igraph.Graph.

    Notes
    -------
    Branching points (vertices with more than two neighbors) are the new vertices
    The new edges are pairs of the new vertices that are either directly connected in the old graph or that
    are connected by 1 or more vertices in the old graph that have exactly two neighbors.
    """

    #trace skeleton_graph
    #branch points are new vertices
    #edges connect branch points and contain several metrics like surface_area, segment_length, tortuosity, r_avg, r_max, r_min, r_sd, coords
    #Afterwards I can get total length as sum of length of all edges and similar things like surface area etc.


    g = g.copy()

    resolution = np.asarray(resolution, dtype=np.float32)

    #non_branch_points are either end points or slab voxels but not branch points
    non_branch_point_ids = g.vs.select(_degree_lt=3)

    end_point_ids = g.vs.select(_degree=1)
    end_point_ids = set(end_point_ids.indices)

    #split graph at branch points -> each branch is now a separate connected component in the graph
    #Attention: Vertices have new indices in this subgraph
    g_branches = g.induced_subgraph(non_branch_point_ids)

    # get list of lists: each list contains the vertex ids of a single branch
    branches = g_branches.connected_components()
    branches = list(branches)

    num_branches = len(branches)

    print(f"Start processing {num_branches} branches.")

    #Then do feature extraction on branches
    #Be careful g_branches has other indices than g
    features, edges = extract_branches_features(g, g_branches, branches, non_branch_point_ids, end_point_ids, resolution)


    #There can also be some edges between branch points which need to be handled
    #IMPORTANT: This was necesarry for an earlier version of our algorithm, where branch point cliques were resolved in
    # a different fashion. Currently "extract_bp_branch_features(...)" will not return any additional edges or features
    branch_point_ids = g.vs.select(_degree_gt=2)
    g_bp = g.induced_subgraph(branch_point_ids)
    bp_branches = g_bp.es()

    num_bp_branches = len(bp_branches)

    #print(f"Start processing {num_bp_branches} branches only consisting of branchpoints connections.")

    # Then do feature extraction on b_branches
    # Be careful g_bp has other indices than g
    # bp stands for branch point
    b_features, b_edges = extract_bp_branches_features(g, bp_branches, branch_point_ids, resolution)
    features.extend(b_features)
    edges.extend(b_edges)

    return create_reduced_graph(g, features, edges)


# Code modified from:
######################################################################################
#   Title: small_seg_path
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/feature_extraction.py
######################################################################################
def trace_small_branch(g, points, non_branch_point_ids):
    """ Trace branches of length 1.

    Parameters
    ----------
    g : ig.Graph
        A non-reduced but cleaned graph generated from a skeleton
    points : list of int
        A list that contains the ids of vertices belonging to the given branch. In case of small branch only list of one int.
        The ids are from another graph that does not include branch points (points with more than two neighbors).
    non_branch_point_ids : igraph.VertexSeq
        A VertexSeq object that maps the points ids to vertex ids in the non-reduced but cleaned graph g.

    Returns
    -------
    extended_points : list of ints
        A list of vertex ids in g that represent the ordered branch path including the branch points (points with more than two neighbors)
    """

    vert = non_branch_point_ids[points[0]].index

    # Add branch point to the segment
    extended_points = g.neighbors(vert)
    extended_points.insert(1, vert)  # Insert vert into middle or if it is an endpoint (has only one neighbor) to the right

    return extended_points


# Code modified from:
######################################################################################
#   Title: large_seg_path
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/feature_extraction.py
######################################################################################
def trace_large_branch(g, g_branches, points, non_branch_point_ids):
    """
    Trace branches of length > 1.
    Parameters
    ----------
    g : ig.Graph
        A non-reduced but cleaned graph generated from a skeleton
    g_branches : ig.Graph
        The graph g but with branching points (vertices with more than two neighbors) removed
    points : list of int
        A list that contains the ids of vertices belonging to one branch.
        The ids are from the g_branches graph that does not include branch points (points with more than two neighbors).
    non_branch_point_ids : igraph.VertexSeq
        A VertexSeq object that maps the points ids to vertex ids in the non-reduced but cleaned graph g.

    Returns
    -------
    point_list : list of ints
        A list of vertex ids in g that represent the ordered branch path including the branch points (points with more than two neighbors)
    """
    degrees = g_branches.degree(points)

    branch_ends = [points[i] for i, d in enumerate(degrees) if d == 1]

    if len(branch_ends) == 2:
        # Find the ordered path of vertices between each endpoint.
        # The indices of this path will be relative
        path = g_branches.get_shortest_paths(branch_ends[0], to=branch_ends[1], output="vpath")[0]

        # Add true indices of our segment path to the point_list.
        point_list = [non_branch_point_ids[point].index for point in path]

        # Extend the point_list by any neighbors on either end of the segment in the original graph.
        end_neighborhood = point_list[0:2] + point_list[-2:]
        for i in range(2):
            for neighbor in g.neighbors(point_list[-i]):  #-0 -> 0 / -1 -> end of array
                if neighbor not in end_neighborhood:
                    if i == 0:
                        point_list.insert(0, neighbor)
                    else:
                        point_list.append(neighbor)

    # Loops in the vasculature
    elif len(branch_ends) != 2:
        point_list = loop_path(g_branches, points, non_branch_point_ids)

    return point_list


# Code modified from:
######################################################################################
#   Title: loop_path
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/feature_extraction.py
######################################################################################
def loop_path(g_branches, points, non_branch_point_ids):
    """
    Trace branches that are closed loops.

    Parameters
    ----------
    g_branches : ig.Graph
        The graph g but with branching points (vertices with more than two neighbors) removed
    points : list of int
        A list that contains the ids of vertices belonging to one branch.
        The ids are from the g_branches graph that does not include branch points (points with more than two neighbors).
    non_branch_point_ids : igraph.VertexSeq
        A VertexSeq object that maps the points ids to vertex ids in the non-reduced but cleaned graph g.

    Returns
    -------
    point_list : list of ints
        A list of vertex ids in g that represent the ordered branch path including the branch points (points with more than two neighbors)
    """

    loop = []
    # Choose a random first point.
    v1 = points[0]
    loop.append(v1)
    previous = v1

    # Now loop through the points until we find the first point.
    looped = False
    i = 0
    size = len(points)
    # Start with random point in the segment.
    while not looped:
        if i > size:  # Safeguard
            looped = True
            break
        # Get neighbors of that point. Make sure the neighbors don't match the 2nd to previous added.
        # Add them to the loop if they don't.
        ns = g_branches.neighbors(loop[-1])
        if ns[0] != previous and ns[0] != v1:
            loop.append(ns[0])
        elif ns[1] != previous and ns[1] != v1:
            loop.append(ns[1])
        else:
            looped = True
        previous = loop[-2]
        i += 1

    #append the start vertice to end of loop
    loop.append(v1)

    # Convert back into graph units.

    point_list = [non_branch_point_ids[point].index for point in loop]

    return point_list


# Code modified from:
######################################################################################
#   Title: small_seg_path, large_seg_path
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/feature_extraction.py
######################################################################################
def extract_branches_features(g, g_branches, branches, non_branch_point_ids, end_point_ids, resolution):
    """Trace branches + extract features from these branches.

    Parameters
    ----------
    g : ig.Graph
        A non-reduced but cleaned graph generated from a skeleton
    g_branches : ig.Graph
        The graph g but with branching points (vertices with more than two neighbors) removed
    branches : ig.VertexClustering
        A VertexClustering that clusters vertices into their branches.
    non_branch_point_ids : igraph.VertexSeq
        A VertexSeq object that maps the vertex ids from the branches to vertex ids in the non-reduced but cleaned graph g.
    end_point_ids : set of in the non-reduced but cleaned graph g
        The vertex ids of endpoints in the
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Returns
    -------
    features : list of FeatureSet
        A list containing the features for every branch.
    edges : list of lists
        A list containing pairs of branchpoint vertex ids (the new vertices in the reduced graph) from the original
        graph g that are connected by a branch.
    """

    features = []
    edges = []

    for branch in branches:

        if len(branch) == 1:
            ordered_branch = trace_small_branch(g, branch, non_branch_point_ids)

        elif len(branch) > 1:
            ordered_branch = trace_large_branch(g, g_branches, branch, non_branch_point_ids)
        else:
            raise RuntimeError("The graph must not contain branches of lenght 0!")

        edges.append([ordered_branch[0], ordered_branch[-1]])

        f = extract_features(g, ordered_branch, end_point_ids, resolution)
        features.append(f)

    return features, edges

# Code modified from:
######################################################################################
#   Title: branch_segment_feature_extraction
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/feature_extraction.py
######################################################################################
def extract_bp_branches_features(g, branchpoint_branches, branch_point_ids, resolution):
    """
    Trace branches that only consist of branchpoints (vertices with more than 2 neighbors) and extract their features.
    Parameters
    ----------
    g : ig.Graph
        A non-reduced but cleaned graph generated from a skeleton
    branchpoint_branches : ig.EdgeSeq
        A edge sequence that contains all the direct connections between branchpoints.
    branch_point_ids : igraph.VertexSeq
        A VertexSeq object that maps the vertex ids from the branches to vertex ids in the non-reduced but cleaned graph g.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Returns
    -------
    features : list of FeatureSet
        A list containing the features for every branch.
    edges : list of lists
        A list containing pairs of branchpoint vertex ids (the new vertices in the reduced graph) from the original
        graph g that are connected by a branch.
    """
    features = []
    edges = []

    for branch in branchpoint_branches:
        ends = [branch_point_ids[branch.target].index, branch_point_ids[branch.source].index]

        edges.append(ends)

        f = extract_features(g, ends, set(), resolution)
        features.append(f)

    return features, edges


# Code modified from:
######################################################################################
#   Title: feature_extraction
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/feature_extraction.py
######################################################################################

FeatureSet = namedtuple("FeatureSet", ['volume', 'volume_med', 'surface_area', 'surface_area_med', 'length', 'tortuosity', 'radius_avg', 'radius_med', 'radius_max', 'radius_min', 'radius_sd', 'radii', 'coords', 'is_end_branch'])


def extract_features(g, points, global_end_point_ids, resolution):
    """Extract the features from branches.
    Parameters
    ----------
    g : ig.Graph
        A non-reduced but cleaned graph generated from a skeleton
    points : list of int
        A list that contains the ids of the ordered vertices belonging to one branch.
    global_end_point_ids : list of int, None
        Either a list of global endpoint in g (vertices that have exactly one neighbor) or
        None if we are sure that points does not contain endpoints
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]

    Returns
    -------
    features : FeatureSet
        A namedtuple containing the features.
    """

    is_end_branch = False

    for p in points:
        if p in global_end_point_ids:
            is_end_branch = True
            break

    v_seq = g.vs.select(points)

    radii = v_seq["radius"]

    r_mean = np.mean(radii)
    r_med = np.median(radii)
    r_max = np.max(radii)
    r_min = np.min(radii)
    r_sd = np.std(radii)

    #coordinates included in branch
    coords = v_seq["coord"]
    coords = np.array(coords)

    #length of branch
    deltas = coords[:-1] - coords[1:]
    squares = (deltas * resolution) ** 2
    path_lengths = np.sqrt(np.sum(squares, axis=1))
    branch_length = np.sum(path_lengths)

    #Tortuosity (branch length divided by the euclidean distance between the start and end points)
    delta = coords[-1] - coords[0]
    square = (delta * resolution) ** 2
    endpoints_dist = np.sqrt(np.sum(square))

    if abs(branch_length-endpoints_dist) < 0.0001:
        tortuosity = 1

    elif(endpoints_dist > np.min(resolution)):
        tortuosity = branch_length / endpoints_dist

    else: #for loops in vasculature we define turtuosity like this (VesselVio set this to 0 instead)
        tortuosity = branch_length / np.min(resolution)


    #Surface Area Estimate
    # Only lateral surface area, not of caps
    surface_area = 2 * np.pi * r_mean * branch_length
    surface_area_med = 2 * np.pi * r_med * branch_length

    #Volume Estimate
    volume = np.pi * r_mean ** 2 * branch_length
    volume_med = np.pi * r_med ** 2 * branch_length

    features = FeatureSet(
        volume,
        volume_med,
        surface_area,
        surface_area_med,
        branch_length,
        tortuosity,
        r_mean,
        r_med,
        r_max,
        r_min,
        r_sd,
        radii,
        coords,
        is_end_branch
    )

    return features


# Code modified from:
######################################################################################
#   Title: simplify_graph
#   Authors: Jacob Bumgarner
#   Date: 12th May 2023
#   Availability: https://github.com/JacobBumgarner/VesselVio/blob/main/library/graph_processing.py
######################################################################################
def create_reduced_graph(g, features, edges):
    """Create the reduced graph from the non-reduced graph g, the features of all edges and the edges (pairs of vertex ids) itself.

    Parameters
    ----------
    g : ig.Graph
        A non-reduced but cleaned graph generated from a skeleton
    features : list of FeatureSet
        The features belonging to edges.
    edges : list of lists of shape (2,)
        A list that contains pairs of vertex ids from g.
        These vertex ids are branchpoints (vertices with more than 2 neighbors) that are connected by branches.

    Returns
    -------
    g : igraph.Graph
        The reduced graph.
    """

    branch_count = len(features)
    volumes = np.zeros(branch_count)
    volumes_med = np.zeros(branch_count)
    surface_areas = np.zeros(branch_count)
    surface_areas_med = np.zeros(branch_count)
    lengths = np.zeros(branch_count)
    tortuosities = np.zeros(branch_count)
    radii_avg = np.zeros(branch_count)
    radii_med = np.zeros(branch_count)
    radii_max = np.zeros(branch_count)
    radii_min = np.zeros(branch_count)
    radii_sd = np.zeros(branch_count)
    coords_lists = []
    radii_lists = []
    is_end_branch_list = []

    for i, feature in enumerate(features):
        volumes[i] = feature.volume
        volumes_med[i] = feature.volume_med
        surface_areas[i] = feature.surface_area
        surface_areas_med[i] = feature.surface_area_med
        lengths[i] = feature.length
        tortuosities[i] = feature.tortuosity
        radii_avg[i] = feature.radius_avg
        radii_med[i] = feature.radius_med
        radii_max[i] = feature.radius_max
        radii_min[i] = feature.radius_min
        radii_sd[i] = feature.radius_sd


        coords_lists.append(feature.coords)
        radii_lists.append(feature.radii)
        is_end_branch_list.append(feature.is_end_branch)

    g.delete_edges(g.es())
    g.add_edges(
        edges,
        {
            "volume": volumes,
            "volume_med": volumes_med,
            "surface_area": surface_areas,
            "surface_area_med": surface_areas_med,
            "length": lengths,
            "tortuosity": tortuosities,
            "radius_avg": radii_avg,
            "radius_med": radii_med,
            "radius_max": radii_max,
            "radius_min": radii_min,
            "radius_sd": radii_sd,
            "coords_list": coords_lists,
            "radii_list": radii_lists,
            "is_end_branch": is_end_branch_list
        },
    )
    g.delete_vertices(g.vs.select(_degree=0))

    return g


def prune_short_end_branches(reduced_g, keep_above=2.0):
    """ Remove short end branches.

    Parameters
    ----------
    reduced_g : ig.Graph
        A reduced graph.
    keep_above: int, float
        Branches that are ending in an endpoint (vertex with only one neighbor)
        and that are shorter than or equal to keep_above micrometers are removed

    Returns
    -------
    reduced_g : ig.Graph
        The reduced graph, where short endbranches are removed.
    """

    end_branches = reduced_g.es.select(is_end_branch_eq=True)

    ids = np.asarray(end_branches.indices)

    end_branches_length = end_branches["length"]
    end_branches_length = np.asarray(end_branches_length)

    delete = end_branches_length <= keep_above

    delete = ids[delete]

    print(f"Edges before: {len(reduced_g.es)}")
    reduced_g.delete_edges(delete)
    print(f"Edges after pruning: {len(reduced_g.es)}")

    print(f"Vertices before: {len(reduced_g.vs)}")
    non_isolated = reduced_g.vs.select(_degree_gt=0)
    reduced_g = reduced_g.induced_subgraph(non_isolated)
    print(f"Vertices after pruning: {len(reduced_g.vs)}")


    return reduced_g


def perform_graph_generation(segmentation_input_folder_path, output_folder_path, resolution, save_graph=True, canceled=Event(), progress_status=None):
    """ Generate a graph from binary segmentation images.

    Parameters
    ----------
    segmentation_input_folder_path : str, pathlib.Path
        The path to the folder containing the binary image slices to be processed.
    output_folder_path : str, pathlib.Path
        The path to the folder where the 3D skeletonization result is stored.
    resolution : (3,) array, 3-tuple
        Array, that specifies resolution in z, y, x direction in order [z_resolution, y_resolution, x_resolution]
    save_graph : bool
        A boolean that determines whether the igraph.Graph should be stored via pickle.
    canceled : threading.Event, optional
        An event object that allows to cancel the process
    progress_status : PyQt6.QtCore.pyqtSignal
        A pyqtSignal that can be used to emit the current progress status

    Notes
    ----------
    This method writes important graph statistics to a txt file that is stored in the graph folder within output_folder_path
    """

    if canceled.is_set():
        return

    output_folder_path = Path(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    time_str = time.strftime("%H%M%S_")
    skel_out_folder_name = time_str + "skeleton_output"
    skel_output_folder = output_folder_path / skel_out_folder_name
    os.makedirs(skel_output_folder, exist_ok=True)

    progress = 0

    if progress_status is not None:
        progress_status.emit(["Generate skeleton...", progress])

    start = timer()

    skeletonize3D(segmentation_input_folder_path, skel_output_folder, resolution, canceled=canceled)

    end = timer()
    print(f"Skeletonization took {end - start} seconds.")

    progress = 30
    if canceled.is_set():
        return

    if progress_status is not None:
        progress_status.emit(["Generate graph from skeleton...", progress])

    start = timer()

    g = generate_skeleton_graph(skel_output_folder, segmentation_input_folder_path, resolution)


    #Create graph output directory
    graph_out_folder_name = time_str + "graph_output"
    graph_output_folder = output_folder_path / graph_out_folder_name
    os.makedirs(graph_output_folder, exist_ok=True)


    # save graph via pickle
    # graph can be read again by my_graph = ig.Graph.Read_Pickle(path_to_pickle_file)
    if save_graph:
        out_file_name = time_str + "skeleton_igraph(before_clique_removal).pickle"
        output_file_path = graph_output_folder / out_file_name
        g.write_pickle(output_file_path)

    progress = 50
    if canceled.is_set():
        return

    if progress_status is not None:
        progress_status.emit(["Clean graph...", progress])

    g = clean_graph(g)

    progress = 60
    if canceled.is_set():
        return

    if progress_status is not None:
        progress_status.emit(["Reduce graph...", progress])

    g = reduce_graph(g, resolution)

    progress = 70

    if canceled.is_set():
        return

    if progress_status is not None:
        progress_status.emit(["Prune...", progress])

    g = prune_short_end_branches(g, keep_above=2.0)

    end = timer()
    print(f"Graph generation took {end - start} seconds.")

    progress = 90
    if progress_status is not None:
        progress_status.emit(["Save statistics...", progress])


    out_file_name = time_str + "graph_statistics.txt"
    output_file_path = graph_output_folder / out_file_name

    reduced_graph_stats_summary(g, write_summary_to_file=True, output_file_path=output_file_path)

    #save graph via pickle
    #graph can be read again by my_graph = ig.Graph.Read_Pickle(path_to_pickle_file)
    if save_graph:
        out_file_name = time_str + "reduced_igraph.pickle"
        output_file_path = graph_output_folder / out_file_name
        g.write_pickle(output_file_path)

    progress = 100
    if progress_status is not None:
        progress_status.emit(["Finished...", progress])


def reduced_graph_stats_summary(g, write_summary_to_file=True, output_file_path=None):
    """ Write the statistics of the reduced gaph g to a txt file

    Parameters
    ----------
    g : ig.Graph
        A reduced graph.
    write_summary_to_file : bool
        Whether to write the statistics to a txt file or to print them to std.out
    output_file_path : str, pathlib.Path, optional
        The path where the txt file with the statistics is stored.
    """

    edges = g.es

    volumes = edges["volume"]
    volumes_med = edges["volume_med"]
    surface_areas = edges["surface_area"]
    surface_areas_med = edges["surface_area_med"]
    lengths = edges["length"]
    tortuosities = edges["tortuosity"]
    radii_avg = edges["radius_avg"]

    total_volume_from_means = np.sum(volumes)
    total_volume_from_med = np.sum(volumes_med)
    total_surface_area_from_means = np.sum(surface_areas)
    total_surface_area_from_med = np.sum(surface_areas_med)

    total_length = np.sum(lengths)

    average_tortuosity = np.mean(tortuosities)
    average_radius = np.mean(radii_avg)

    out_str = "------------------------------------"
    out_str += "\n"
    out_str += "Average vessel segment radius [" + u"\u03bc" + "m]: " + str(average_radius)
    out_str += "\n"
    out_str += "Total length of vasculature [" + u"\u03bc" + "m]: " + str(total_length)
    out_str += "\n"
    out_str += "Total vasculature volume [" + u"\u03bc" + "m^3]: " + str(total_volume_from_means)
    #out_str += "\n"
    #out_str += "Total volume (from branch radius medians) [" + u"\u03bc" + "m^3]: " + str(total_volume_from_med) #Median is close to mean -> no benefit in also stating this
    #out_str += "\n"
    #out_str += "Total surface area (from branch radius means) [" + u"\u03bc" + "m^2]: " + str(total_surface_area_from_means) # not needed currently
    #out_str += "\n"
    #out_str += "Total surface area (from branch radius medians) [" + u"\u03bc" + "m^2]: " + str(total_surface_area_from_med) # not needed currently
    #out_str += "\n"
    #out_str += "Average tortuosity: " + str(average_tortuosity) # not needed currently
    out_str += "\n"
    out_str += "------------------------------------"

    if not write_summary_to_file:
        print(out_str)

    else:
        if output_file_path is None:
            time_str = time.strftime("%H%M%S_")
            out_name = time_str + "graph_statistics.txt"
            output_file_path = os.path.join(OUTPUT_DIRECTORY, out_name)

        with open(output_file_path, 'w', encoding="utf-8") as f:
            f.write(out_str)





#########################################################Tests##########################################################
#File names need to be adjusted for some of the tests if you want to execute them on your machine on specific graphs
########################################################################################################################

def skeleton_coords_from_reduced_graph(g, sort_z=True):
    """ Get all 3D coordinates from a reduced graph.

    Parameters
    ----------
    g : ig.Graph
        A reduced graph.
    sort_z : bool
        Determines whether to sort the returned coordinates list after the z coordinates.

    Returns
    -------
    coordinates : np.ndarray
        The list of coordinates representing the graph.
    """

    coordinates = g.es["coords_list"]
    coordinates = np.concatenate(coordinates, axis=0)
    coordinates = coordinates.astype(np.uint16)
    coordinates = np.unique(coordinates, axis=0)

    if sort_z:
        coordinates = coordinates[coordinates[:, 0].argsort()]

    return coordinates


def skeleton_coords_from_skel_graph(g, sort_z=True):
    """ Get all 3D coordinates from a reduced graph.

    Parameters
    ----------
    g : ig.Graph
        A reduced graph.
    sort_z : bool
        Determines whether to sort the returned coordinates list after the z coordinates.

    Returns
    -------
    coordinates : np.ndarray
        The list of coordinates representing the graph.
    """

    coordinates = g.vs["coord"]
    coordinates = np.asarray(coordinates)
    coordinates = coordinates.astype(np.uint16)
    #i = 0
    #for c in coordinates:
        #a = np.array([21, 1216, 1638], dtype=np.uint16)
        #if (c == a).all():
            #print(i)
        #i+=1

    coordinates = np.unique(coordinates, axis=0)

    if sort_z:
        coordinates = coordinates[coordinates[:, 0].argsort()]

    return coordinates


def write_reduced_graph_to_tiffs(g, shape, filename, output_folder_path):
    """Write a reduced graph to image slices. This is basically the cleaned and reduced skeleton.

    Parameters
    ----------
    g : ig.Graph
        A reduced graph.
    shape : 3-tuple, array of shape (3,)
        The shape of the image stack to which the graph should be written.
    filename : str
        The filename of the output images. The Z coordinate is always appended to the filename.
    output_folder_path : str, pathlib.Path
        The path to the folder where the output images are stored.
    """

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    coords = skeleton_coords_from_reduced_graph(g, sort_z=True)

    prev_z = coords[0,0]

    #fill gap from 0th slice until first slice with skeleton point
    write_background_slices(0, prev_z, shape, filename, output_folder)

    img = np.zeros(shape[1:], dtype=np.uint8)

    for c in coords:
        z = c[0]

        if z != prev_z:
            outpath = filename + "Z" + str(prev_z).zfill(4) + ".ome.tif"
            outpath = output_folder / outpath
            save_image(img, outpath, dtype=np.uint8)

            img = np.zeros(shape[1:], dtype=np.uint8)
            #fill gap between z slices
            if z-prev_z > 1:
                write_background_slices(prev_z+1, z, shape, filename, output_folder)
            prev_z = z

        img[c[1], c[2]] = 255

    outpath = filename + "Z" + str(z).zfill(4) + ".ome.tif"
    outpath = output_folder / outpath
    save_image(img, outpath, dtype=np.uint8)

    #fill gap until end
    write_background_slices(z+1, shape[0], shape, filename, output_folder)


def write_skel_graph_to_tiffs(g, shape, filename, output_folder_path):
    """Write a reduced graph to image slices. This is basically the cleaned and reduced skeleton.

    Parameters
    ----------
    g : ig.Graph
        A reduced graph.
    shape : 3-tuple, array of shape (3,)
        The shape of the image stack to which the graph should be written.
    filename : str
        The filename of the output images. The filename is always prepended by the Z coordinate.
    output_folder_path : str, pathlib.Path
        The path to the folder where the output images are stored.
    """

    output_folder = Path(output_folder_path)
    os.makedirs(output_folder, exist_ok=True)

    coords = skeleton_coords_from_skel_graph(g, sort_z=True)

    prev_z = coords[0, 0]

    # fill gap from 0th slice until first slice with skeleton point
    write_background_slices(0, prev_z, shape, filename, output_folder)

    img = np.zeros(shape[1:], dtype=np.uint8)

    for c in coords:
        z = c[0]

        if z != prev_z:
            outpath = filename + "Z" + str(prev_z).zfill(4) + ".ome.tif"
            outpath = output_folder / outpath
            save_image(img, outpath, dtype=np.uint8)

            img = np.zeros(shape[1:], dtype=np.uint8)
            # fill gap between z slices
            if z - prev_z > 1:
                write_background_slices(prev_z + 1, z, shape, filename, output_folder)
            prev_z = z

        img[c[1], c[2]] = 255

    outpath = filename + "Z" + str(z).zfill(4) + ".ome.tif"
    outpath = output_folder / outpath
    save_image(img, outpath, dtype=np.uint8)

    # fill gap until end
    write_background_slices(z + 1, shape[0], shape, filename, output_folder)


def write_background_slices(start, end, shape, filename, output_folder):
    """Write black images (all pixels 0) for z-slices in interval [start,end)"""
    for z in range(start, end):

        img = np.zeros(shape[1:], dtype=np.uint8)
        outpath = filename + "Z" + str(z).zfill(4) + ".ome.tif"
        outpath = output_folder / outpath
        save_image(img, outpath, dtype=np.uint8)



def test_skeleton_graph():

    g = ig.Graph.Read_Pickle(r"..\data\graph\g1.pickle")

    skeleton_volume = np.load(r"../../data/skeleton_lee.npy")

    points = np.where(skeleton_volume)
    points = np.stack(points, axis=-1)
    points = points.astype(np.uint16)

    assert len(g.vs) == len(points)


    s = g.copy()

    s.simplify(multiple=True, loops=True)

    assert len(s.es) == len(g.es)

    radii = g.vs["radius"]

    print(f"Max radius: {np.max(radii)}")

    print(f"Min radius: {np.min(radii)}")

    print(f"Mean radius: {np.mean(radii)}")

    print(f"Median radius: {np.median(radii)}")


def test_clean_graph():

    g = ig.Graph.Read_Pickle(r"..\data\graph\g1.pickle")

    g_clean = ig.Graph.Read_Pickle(r"..\data\graph\g1_clean.pickle")

    assert len(g.es) >= len(g_clean.es)

    assert len(g.vs) >= len(g_clean.vs)


def test_reduced_graph(g=None):
    if g is None:
        g = ig.Graph.Read_Pickle(r"..\data\graph\g1_reduced_pruned.pickle")

    for branch in g.es:
        radii = branch["radii_list"]
        coords = branch["coords_list"]

        r_max = np.max(radii)
        r_min = np.min(radii)
        r_mean = np.mean(radii)
        r_std = np.std(radii)

        assert r_max == branch["radius_max"]
        assert r_min == branch["radius_min"]
        assert r_mean == branch["radius_avg"]
        assert r_std == branch["radius_sd"]

        #coords = branch["coords_list"]
        #length = branch["length"]

        #if len(coords) < 4 and length > 10:
            #debug = 1


def test_reduced_graph_coords(g=None, res=(2.0,0.325,0.325)):
    if g is None:
        g = ig.Graph.Read_Pickle(r"..\data\graph\g1_reduced.pickle")

    maxDelta = 0
    for branch in g.es:
        coords = branch["coords_list"]
        length = branch["length"]

        deltas = coords[:-1] - coords[1:]
        squares = (deltas * res)
        squares = squares ** 2
        path_lengths = np.sqrt(np.sum(squares, axis=1))
        branch_length = np.sum(path_lengths)

        assert abs(length-branch_length) < 0.0001

        mdelta = np.max(deltas)

        if mdelta > maxDelta:
            maxDelta = mdelta

    print(f"Maximal distance between two neighboring branch coords: {maxDelta}")


def test_reduced_graph_vertices():

    g_cleaned = ig.Graph.Read_Pickle(r"..\data\graph\g1_clean.pickle")
    g_reduced = ig.Graph.Read_Pickle(r"..\data\graph\g1_reduced.pickle")

    degrees = np.asarray(g_cleaned.vs.degree())
    ids = np.asarray(g_cleaned.vs.indices)

    branch_points = degrees > 2
    end_points = degrees == 1

    non_slab_points = np.logical_or(branch_points, end_points)
    slab_points = np.logical_not(non_slab_points)

    non_slab_points = ids[non_slab_points]
    slab_points = ids[slab_points]

    num_branch_points = np.count_nonzero(branch_points)
    num_end_points = np.count_nonzero(end_points)
    num_slab_voxels = len(slab_points)
    print(f"Skeleton graph has {num_branch_points} branch points, {num_end_points} endpoints and {num_slab_voxels} slab voxels.")
    print(f"Total number of skeleton points: {num_branch_points + num_slab_voxels + num_end_points}")

    ns_coords = g_cleaned.vs[non_slab_points]["coord"]
    ns_coords = np.asarray(ns_coords, dtype=np.uint16)
    s_coords = g_cleaned.vs[slab_points]["coord"]
    s_coords = np.asarray(s_coords, dtype=np.uint16)

    ns_set = set()
    for c in ns_coords:
        key = str(c[0]) + "_" + str(c[1]) + "_" + str(c[2])
        ns_set.add(key)

    s_set = set()
    for c in s_coords:
        key = str(c[0]) + "_" + str(c[1]) + "_" + str(c[2])
        s_set.add(key)



    # Test whether all vertices of the reduced gaph are branch points in the cleaned graph
    reduced_vs_coords = np.asarray(g_reduced.vs["coord"]).astype(np.uint16)

    for c in reduced_vs_coords:
        elem = str(c[0]) + "_" + str(c[1]) + "_" + str(c[2])
        assert elem in ns_set, f"The vertex coordinate {c} is not a branch or endpoint in the skeleton graph."


    # Test whether middle coords are unique and slab_points

    reduced_es_coords_lists = g_reduced.es["coords_list"]
    slab_coords = []

    for c_list in reduced_es_coords_lists:
        if len(c_list) < 3: #branch point is at index 0 / end point is at index 1
            continue
        else:
            slab_coords.extend(c_list[1:-1])

    slab_coords = np.asarray(slab_coords, dtype=np.uint16)

    for c in slab_coords:
        elem = str(c[0]) + "_" + str(c[1]) + "_" + str(c[2])
        assert elem in s_set, f"The coordinate {c} is not a slab voxel in the skeleton graph."


    degrees = g_reduced.vs.degree()
    degrees = np.asarray(degrees)

    end_points = degrees == 1
    num_end_points = np.count_nonzero(end_points)
    branch_points = degrees > 2
    num_branch_points = np.count_nonzero(branch_points)
    num_slab_voxels = len(slab_coords)

    print(f"Reduced graph has {num_branch_points} branch points, {num_end_points} endpoints and {num_slab_voxels} slab voxels.")
    print(f"Total number of skeleton points: {num_branch_points + num_slab_voxels + num_end_points}")
