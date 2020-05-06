import numpy as np
import nibabel as nib
import os
import sys
import pickle
import multiprocessing
import math

from .CTH_seg_common import data
from pyCudaImageWarp import augment3d, cudaImageWarp, scipyImageWarp

# Ensure that Tensorflor has not yet been imported
def tf_guard():
    # Check that Tensorflow has not yet been imported! It's badly behaved!
    try:
        tf
    except:
        return;

    raise ValueError('Must not import tensorflow prior to creating an Inferer!')

# Get the dtype and shape of a batch, given a list of tensors
def __batch_info__(tensorList):
   firstTensor = tensorList[0] 
   dtype = firstTensor.dtype
   element_shape = tuple([int(x) for x in firstTensor.shape])
   batch_shape = (len(tensorList),) + element_shape

   return dtype, batch_shape

"""
A class which uses the Tensorflow model in inference-only mode.
"""
class Inferer:
    def __init__(self, pb_path, params_path):
        """
            Loads the model from a file.
            Arguments:
                pb_path - Path to the serialized .pb file
                params_path - Path to the .pkl meta-parameter file
        """

        # Import tensorflow.
        tf_guard()
        try:
            # Compatibility with TF 2.0--this is TF 1 code
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        except:
            import tensorflow as tf
        from tensorflow.python.platform import gfile
        from tensorflow.python.client import device_lib

        # Start a new graph. This will not necessarily protect us against the
        # new graph having tensors of the same name. Use isolate_inference for
        # this.
        with tf.Graph().as_default():
            
            # Unpack the network parameters
            with open(params_path, 'rb') as f:
                self.params = pickle.load(f)

            # Load the frozen graph
            with gfile.FastGFile(pb_path,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # Query the available GPUs
            local_device_protos = device_lib.list_local_devices()
            gpuNames = [x.name for x in local_device_protos if x.device_type == 'GPU']

            # Initialize on each gpu
            xList = []
            probList = []
            for gpuName in gpuNames:
                with tf.device(gpuName):
                    device_name, device_num = gpuName.strip('/device:').strip('/').split(':')
                    with tf.name_scope(device_name + device_num) as scope:

                        # Initialize the graph, extract the input and output nodes
                        X, prob = tf.import_graph_def(
                            graph_def,
                            name='',
                            return_elements=['X:0', 'prob:0']
                        )

                xList.append(X)
                probList.append(prob)
            self.xList = xList

            # Create a tensor for the full (batch) output
            self.prob = tf.concat(probList, axis=0)

            # Disable Tensorflow auto-tuning, which is ridiculously slow
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

            # Configure Tensorflow to only use as much GPU memory as it actually needs
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            # Spawn a session
            self.session = tf.Session(config=config)

    def tile_inference(self, vol, labels=None, scale_factors=None, 
        oob_image_val=0, oob_label=0, device=None, api='cuda'):
        """
            Calls tile_inference() with the right Tensorflow objects.
        """
        return tile_inference(vol, self.xList, self.prob, self.params, 
                self.session, labels=labels, scale_factors=scale_factors, 
                device=device, api=api)

def tile_inference(vol, vol_ph_list, prob_ph, params, session, feed_dict={},
        labels=None, labels_ph=None, weights_ph=None, loss_ph=None, 
        voronoi=None, voronoi_ph=None, num_objects_ph=None,
        fuzzy_labels=None, fuzzy_labels_ph=None,
        fuzzy_weights=None, fuzzy_weights_ph=None,
        scale_factors=None, oob_image_val=0, oob_label=0, device=None, 
        api='cuda'):
    """
        Run inference, using cropping and tiling to cover the whole image
        Inputs:
            vol - the image the volume
            vol_ph_list - a list of Tensorflow placeholders for the image data. If this
                has only one element, vol_ph[0] is the batch dimension. (List
                processing is needed for changing the number of GPUs at 
                runtime.)
            prob_ph - a Tensorflow placeholder for the output
            params - metadata about the network, stored in segmentor.params
            session - the Tensorflow session
            feed_dict - any additional tensors which must be fed to the network
            labels - the ground truth labels (Optional)
            labels_ph - a Tensorflow placeholder for the labels (Optional)
            loss_ph - a Tensorflow placeholder for the loss (Optional)
            weights_ph - a Tensorflow placeholder for the weights (Optional)
            voronoi - the voronoi digram (Optional)
            voronoi_ph - a Tensorflow placeholder for the voronoi diagram (Optional)
            num_objects_ph - a Tensorflow placeholder for the number of objects (Optional)
            scale_factors - if provided, stretch the image dimensions by 
                    these factors, with anti-aliasing.
            device - The ID of the CUDA device to use.
            api - Either 'cuda' or 'scipy', for preprocessing
        
        Returns a probability mask of the same shape as vol. Also returns the
        average loss, which is zero unless the labels are provided. Also
        returns the preprocessed image, assembled from tiles
    """

    # Process parameters
    if labels_ph is None and voronoi_ph is not None:
        raise ValueError('Must provide labels_ph to use voronoi_ph')

    # Get the batch size of the network
    batch_size = int(prob_ph.shape[0])

    # Process the input volume placeholders. If there's only one, split it
    # into a new list
    if len(vol_ph_list) < 2:
        vol_ph_list = tf.split(vol_ph_list[0], batch_size, 0)

    # Choose the preprocessing API
    warpApiMap = {
            'cuda': cudaImageWarp,
            'scipy': scipyImageWarp
    }
    warpApi = warpApiMap[api]

    # Do no scaling by default
    if scale_factors is None:
        scale_factors = (1, 1, 1)
    scale_factors = np.array(scale_factors)

    # Pad the image with a channel dimension, if it's missing
    ndim = 3 
    if len(vol.shape) < ndim + 1:
        vol = np.expand_dims(vol, ndim)

    # Get the output size and scaling transform
    scale_factors = np.array(scale_factors)
    space_dims = np.ceil(np.array(vol.shape[:ndim]) * scale_factors).astype(
            int)

    # Get the number of classes
    num_class = int(prob_ph.shape[-1])

    # Get the number of channels
    num_channels = vol.shape[3]

    # Optionally pre-filter the image for anti-aliasing
    for c in range(num_channels):
        vol[:, :, :, c] = data.aaFilter(vol[:, :, :, c], scale_factors)

    # Get the tile info
    crop_dims = np.array(params['data_shape'][:ndim])
    if np.any(crop_dims < space_dims):
        overlap = 0.25 # Percentage of overlap in each dimension
    else:
        overlap = 0.0 # No cropping takes place, no need for overlap
    coverage = 1.0 - overlap
    expansion = 1.0 / coverage
    num_tiles_float = space_dims.astype(float) / crop_dims.astype(float)
    num_tiles = np.ceil(num_tiles_float * expansion).astype(int)
    num_tiles[num_tiles_float <= 1.0] = 1.0
    tiles = []
    tileIdxX, tileIdxY, tileIdxZ = np.meshgrid(
        range(num_tiles[0]), range(num_tiles[1]), range(num_tiles[2])
    )
    tiles = np.array([tileIdxX.flatten(), tileIdxY.flatten(), 
        tileIdxZ.flatten()])

    # Split the tiles into batches
    num_tiles = tiles.shape[1]
    num_batches = int(math.ceil(float(num_tiles) / batch_size))
    batch_inds = [
        range(batch_size * i, min(batch_size * (i + 1), num_tiles)) 
            for i in range(num_batches)
    ]

    # Run the model on overlapping tiles
    valid = (labels != -1 if labels is not None 
            else np.ones(space_dims, dtype=bool))
    counts = np.zeros(space_dims, dtype=int)
    vol_proc = np.zeros(tuple(space_dims) + (num_channels,))
    loss = 0.0
    loss_iter = 0
    pred = np.zeros(tuple(space_dims) + (num_class,), dtype=np.float32)
    for inds in batch_inds:

        cropSlices = []
        tileSlices = []
        for i in inds:

            # Get the tile starting indices
            tileIdx = tiles[:, i]
            tileOffset = np.minimum(
                    np.floor(tileIdx * crop_dims * coverage).astype(int),
                    np.maximum(space_dims - crop_dims, 0),
            )

            # Get the tile dimensions and slices
            write_shape = np.minimum(space_dims - tileOffset, crop_dims)
            cropSlice = [slice(tileOffset[j], tileOffset[j] + write_shape[j]) \
                for j in range(ndim)]
            tileSlice = [slice(None, write_shape[j]) for j in range(ndim)]

            # Skip this tile if it's completely masked out
            if labels is not None and not np.any(valid[cropSlice]): continue

            # Record the tile dimensions
            cropSlices.append(cropSlice)
            tileSlices.append(tileSlice)

            # Get the cropping translation
            mat_crop = augment3d.get_translation_affine(tileOffset)

            # Add scaling
            mat_scale = np.eye(4)
            mat_scale[:3, :3] = np.diag(1.0 / scale_factors)
            affine = mat_scale.dot(mat_crop)

            # Start pre-processing the image
            warpAffine = affine[:3]
            for c in range(num_channels):
                warpApi.push(
                    vol[:, :, :, c],
                    A=warpAffine,
                    interp='linear',
                    winMin=params['window_min'][c],
                    winMax=params['window_max'][c],
                    shape=crop_dims,
                    oob=oob_image_val,
                    device=device
                )

            # Start warping the labels, if any
            if labels_ph is not None:
                warpApi.push(
                    labels,
                    A=warpAffine,
                    interp='nearest',
                    shape=crop_dims,
                    oob=oob_label,
                    device=device
                )

            # Start warping the voronoi, if any
            if voronoi_ph is not None:
                warpApi.push(
                    voronoi,
                    A=warpAffine,
                    interp='nearest',
                    shape=crop_dims,
                    oob=oob_label,
                    device=device
                )

            # Start warping the fuzzies, if any
            if fuzzy_labels_ph is not None:
                fuzzy_channels = 1 if fuzzy_weights.ndim < 4 else fuzzy_weights.shape[3]
                for c in range(fuzzy_channels):
                    warpApi.push(
                        fuzzy_weights[:, :, :, c],
                        A=warpAffine,
                        interp='linear',
                        shape=crop_dims,
                        oob=0,
                        device=device
                    )
                for c in range(fuzzy_channels):
                    warpApi.push(
                        fuzzy_labels[:, :, :, c],
                        A=warpAffine,
                        interp='nearest',
                        shape=crop_dims,
                        oob=-1,
                        device=device
                    )

        # Initialize the feed dict
        init_list = [(labels_ph, oob_label), 
            (weights_ph, None), (voronoi_ph, None), (num_objects_ph, None),
            (fuzzy_weights_ph, None), (fuzzy_labels_ph, None)]
        for vol_ph in vol_ph_list:
            init_list.append((vol_ph, oob_image_val))
        for ph, val in init_list:
            if ph is None:
                continue
            init_vol = np.zeros([int(x) for x in ph.shape])
            if val is not None:
                init_vol[:] = val
            feed_dict[ph] = init_vol
                
        # Finish pre-processing, assign the inputs
        this_batch_size = len(cropSlices)
        for i in range(this_batch_size):
            # Get the volume
            tile_vol = np.zeros(params['data_shape'], dtype=np.float32)
            for c in range(num_channels):
                tile_vol[:, :, :, c] = warpApi.pop()
            feed_dict[vol_ph_list[i]] = tile_vol[np.newaxis]
            vol_proc[cropSlices[i]] = tile_vol[tileSlices[i]]

            # Optionally get the labels
            if labels_ph is not None:
                tile_labels = warpApi.pop()
                feed_dict[labels_ph][i] = tile_labels

            # Optionally generate the weights
            if weights_ph is not None:
                feed_dict[weights_ph][i] = data.get_weight_map(tile_labels[i])

            # Optionally get the voronoi
            if voronoi_ph is not None:
                tile_voronoi, tile_num_objects = data.reduceLabels(
                    warpApi.pop(), tile_labels >= 0)
                feed_dict[voronoi_ph][i] = tile_voronoi
                feed_dict[num_objects_ph][i] = tile_num_objects
            
            # Optionally get fuzzies
            if fuzzy_weights_ph is not None:
                fuzzy_shape = params['data_shape'][:3] + (fuzzy_channels,)
                tile_fuzzy_weights = np.zeros(fuzzy_shape, dtype=np.float32)
                for c in range(fuzzy_channels):
                    tile_fuzzy_weights[:, :, :, c] = warpApi.pop()
                feed_dict[fuzzy_weights_ph][i] = tile_fuzzy_weights
            if fuzzy_labels_ph is not None:
                tile_fuzzy_labels = np.zeros(fuzzy_shape, dtype=np.int32)
                for c in range(fuzzy_channels):
                    tile_fuzzy_labels[:, :, :, c] = warpApi.pop()
                tile_fuzzy_labels, tile_num_objects = data.reduceLabels(
                    tile_fuzzy_labels, tile_labels >= 0)
                feed_dict[fuzzy_labels_ph][i] = tile_fuzzy_labels
                feed_dict[num_objects_ph][i] = tile_num_objects

        # Assign the outputs
        output_ph = (prob_ph,)
        if loss_ph is not None:
            output_ph += (loss_ph,)

        # Run the model
        output_ph = session.run(
                output_ph, 
                feed_dict=feed_dict
        )
        tile_preds = output_ph[0]
        if loss_ph is not None:
            batch_loss = output_ph[1].squeeze()
            if num_objects_ph is not None:
                batch_loss /= np.sum(feed_dict[num_objects_ph])
        assert(np.array_equal(tile_preds.shape[-ndim - 1:-1], crop_dims))

        # Accumulate the output predictions, update the counts
        for i in range(this_batch_size):
            tileSlice = tileSlices[i]
            cropSlice = cropSlices[i]
            pred[cropSlice] += tile_preds[i].squeeze()[tileSlice]
            counts[cropSlice] += 1

        # Optionally accumulate the loss
        if loss_ph is not None and not np.isnan(batch_loss):
            loss += batch_loss
            loss_iter += 1

    # Average the accumulated predictions
    pred /= np.expand_dims(np.maximum(counts, 1), -1)
    loss /= max(loss_iter, 1)

    return pred, loss, vol_proc

"""
    Main function for the child process, controlling all TF stuff
"""
def __child_main__(pb_path, params_path, inputQ, outputQ):

    try:
        # Initialize TF stuff
        print('Initializing inferer (child)')
        inferer = Inferer(pb_path, params_path)
        print('Initialized inferer (child)')

        # Listen for requests
        while True:
            # Wait for a request
            print('Waiting for inputs (child)')
            input_args = inputQ.get()

            if input_args is None:
                # Signal to finish
                print('Exiting (child)')
                return
            else:
                vol, scale_factors, device, api = input_args


            # Process the request
            print('Processing (child)')
            outputs = inferer.tile_inference(vol, scale_factors, device=device, 
                    api=api)

            # Return the output
            outputQ.put(outputs)
            print('Returned outputs (child)')

    except Exception as e:
        print('Child failed!')
        outputQ.put(None) # To avoid hanging on error
        raise e

"""
    Like Inferer, but isolated in its own process, to prevent TF bugs
    involved with reusing models.
"""
class IsolatedInferer:
    def __init__(self, pb_path, params_path):

        # Make a pair of Queues for communication
        self.inputQ = multiprocessing.Queue()
        self.outputQ = multiprocessing.Queue()

        # Start the process and initialize the TF model
        self.child = multiprocessing.Process(
            target=__child_main__,
            args=(pb_path, params_path, self.inputQ, self.outputQ)
        )
        self.child.start()

    """
        Runs inference in the child process, returning the result. Recommend 'scipy'
        preprocessing APi to avoid memory issues.
    """
    def tile_inference(self, vol, scale_factors=None, device=None, api='scipy'):
        self.inputQ.put((vol, scale_factors, device, api))
        print('Put arguments (parent)')
        outputs = self.outputQ.get()
        print('Received outputs (parent)')
        return outputs

    """
        Kill (join) the child process.
    """
    def __del__(self):
        print ('Killing child (parent)')
        sys.stdout.flush()
        self.inputQ.put(None)
        self.child.join()

def main():
    """
    Loads the network and performs inference on a single image.
    Usage: 
        python inference.py [network.pb] [params.pkl] [input.nii.gz] [output.nii.gz] [resolution] [class_idx]

    If resolution is provided, resamples the input and output to that 
        resolution. Otherwise, no resampling is performed.
        
    If class_idx is provided, saves the probability of that class. Otherwise, 
        returns a label map via argmax.
    """

    pb_path = sys.argv[1]
    params_path = sys.argv[2]
    nii_in_path = sys.argv[3]
    nii_out_path = sys.argv[4]
    try:
        resolution = float(sys.argv[5]) # Resample to this resolution
    except IndexError:
        resolution = None # No resampling
    try:
        class_idx = sys.argv[6] # In [0, ..., num_class - 1]. Otherwise returns argmax
    except IndexError:
        class_idx = None # Use argmax

    inference_main(pb_path, params_path, nii_in_path, nii_out_path, resolution, class_idx)

def read_nifti(path):
    """
        Utility function to read a Nifti file
    """

    nii = nib.load(path)
    vol = nii.get_data().astype(np.float32)
    units = np.abs(np.diag(nii.header.get_base_affine())[:3])

    return vol, units, nii

def inference_main(pb_path, params_path, nii_in_path, nii_out_path, resolution=None, class_idx=None):
    """
        Entry point for other python scripts.
    """
        
    # Read the Nifti file
    vol, units, nii = read_nifti(path)

    # Call the next level entry point
    inference_main_with_image(pb_path, params_path, vol, units, nii_out_path, 
        nii=nii, 
        resolution=resolution, 
        class_idx=class_idx
    )

def inference_main_with_image(pb_path, params_path, vol, units, nii_out_path, 
        nii=None, resolution=None, class_idx=None): 
    """
        Entry point if the images are already loaded in some other way. 
    """

    # Compute resizing info
    scale_factors = units / resolution if resolution is not None else None
    if np.any(units == 0.0) and resolution is not None:
        raise ValueError("Read invalid units: (%f, %f, %f)" % tuple(units)) 

    # Run inference in a separate process
    pred, loss = isolate_inference(pb_path, params_path, vol, 
        scale_factors=scale_factors)[0:2]

    # Condense to an output volume
    if class_idx is None:
        vol_out = np.argmax(pred, axis=3)
    else:
        vol_out = pred[class_idx]

    # Interpolate the output predictions, if needed
    if scale_factors is not None:
        A = np.zeros((3, 4))
        A[:, 0:3] = np.diag(scale_factors)
        vol_out = cudaImageWarp.warp(
            vol_out,
            A=A,
            shape=vol.shape,
            interp='nearest'
        )

    # Save the result as a Nifti file
    nii_out = nib.Nifti1Image(
        vol_out, 
        nii.affine if nii is not None else np.eye(4), # affine=None gives unintuitive results
        header=nii.header if nii is not None else None
    )
    nib.save(nii_out, nii_out_path)

if __name__ == "__main__":
    main()
