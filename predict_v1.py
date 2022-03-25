import tensorflow as tf
import numpy as np
from glob import glob
import os
from time import time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# timer decorator
def _time(f):
    def wrapper(*args,**kwargs):
        start=time()
        r=f(*args,**kwargs)
        end=time()
        print("%s timed %f" %(f.__name__,end-start))
        return r
    return wrapper

Image.open = _time(Image.open)
np.pad = _time(np.pad)

# tensorflow GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GP0U
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# DeepLab Model structure
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = tf.keras.Input(shape=(image_size, image_size, 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)
    input_a = tf.keras.layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = tf.keras.layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = tf.keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)

@_time
def reshape_split(image:np.ndarray,kernel_size:tuple):
    img_height,img_width,channels=image.shape
    tile_height,tile_width = kernel_size
    tiled_array = image.reshape(img_height//tile_height,
                                tile_height,
                                img_width//tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1,2)
    return tiled_array

@_time
def classifyDL_v2(tiles):
    wsiavg = np.zeros_like(tiles).astype(np.float16)
    wsiavg = wsiavg[:,:,:,:,0]
    wsialone = np.zeros_like(wsiavg)
    wsiavg = np.repeat(wsiavg[:,:,:,:,np.newaxis],13,axis=4) #pre-allocate probability map

    for idx,row in enumerate(tiles):
        print('row: ',idx+1,'/',len(tiles))
        pred_dataset = tf.data.Dataset.from_tensor_slices(row) #this function is only for small dataset fucking hell; only works for a row of image.
        pred_dataset = pred_dataset.batch(4, drop_remainder=False)

        predictions1 = model1.predict(pred_dataset)
        predictions2 = model2.predict(pred_dataset)
        predictions3 = model3.predict(pred_dataset)
        predictions4 = model4.predict(pred_dataset)
        predictions5 = model5.predict(pred_dataset)

        predictions1 = np.squeeze(predictions1)
        predictions2 = np.squeeze(predictions2)
        predictions3 = np.squeeze(predictions3)
        predictions4 = np.squeeze(predictions4)
        predictions5 = np.squeeze(predictions5)

        prediction_avg = np.average(np.stack([predictions1,predictions2,predictions3,predictions4,predictions5]),axis=0)
        wsiavg[idx] = prediction_avg
        wsialone[idx] = np.argmax(prediction_avg, axis=3).astype('uint8')
    return wsiavg,wsialone

#stitch tiles into wsi
def stitch(tiles,img_height,img_width,img_height2,img_width2,channels):
    wsi = tiles.swapaxes(1,2)
    wsi = wsi.reshape(img_height2,img_width2,channels) #tiles are padded, so use padded image size to stitch
    wsi = wsi[:img_height,:img_width,:] #remove pad
    return np.squeeze(wsi)

IMAGE_SIZE = 1024
NUM_CLASSES = 13
BATCH_SIZE = 6

latest1 = tf.train.latest_checkpoint('fold_1')
model1 = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model1.load_weights(latest1)

latest2 = tf.train.latest_checkpoint('fold_2')
model2 = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model2.load_weights(latest2)

latest3 = tf.train.latest_checkpoint('fold_3')
model3 = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model3.load_weights(latest3)

latest4 = tf.train.latest_checkpoint('fold_4')
model4 = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model4.load_weights(latest4)

latest5 = tf.train.latest_checkpoint('fold_5')
model5 = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model5.load_weights(latest5)

src = r'\\fatherserverdw\Q\research\images\skin_aging\deeplab_trainingset\tif'
dst = r'\\fatherserverdw\Q\research\images\skin_aging\deeplab_trainingset\tif\prediction'
imlist = glob(os.path.join(src,'*.tif'))

# Load image
for impth in imlist:
    base,imnm = os.path.split(impth)

    if os.path.exists(os.path.join(dst, imnm.replace('.tif', 'single.png'))): continue

    imobj = Image.open(os.path.join(src,imnm))
    # Image to Array
    imnp = np.array(imobj)
    imobj.close()
    h,w,_=imnp.shape
    tile_height, tile_width = (1024,1024)
    # Padding
    imnpr = np.pad(imnp, pad_width=[(0, tile_height-h%tile_height),(0, tile_width-w%tile_width),(0, 0)], mode='constant', constant_values=0)
    # imnpr = imnpr / 127.5 - 1 #normalize [-1 1]
    imnpr = imnpr / 255 #normalize [0 1]
    img_height2,img_width2,channels=imnpr.shape
    # Tile
    tiles = reshape_split(imnpr, (1024,1024))
    h2,w2,_=imnpr.shape
    del imnpr

    # Pad and Tile for horz and vert shifts
    # imnphorz = imnp[512:,:,:] #this is actually vertical
    # h_h,h_w,_=imnphorz.shape
    # imnphorzpad =np.pad(imnphorz, pad_width=[(0, tile_height-h_h%tile_height),(0, tile_width-h_w%tile_width),(0, 0)], mode='constant', constant_values=0)
    # imnphorzpad = imnphorzpad / 255
    # tileshorz=reshape_split(imnphorzpad, (1024,1024))
    # h_h2,h_w2,_=imnphorzpad.shape
    # del imnphorz,imnphorzpad

    # imnpvert = imnp[:,512:,:]
    # v_h,v_w,_=imnpvert.shape
    # imnpvertpad =np.pad(imnpvert, pad_width=[(0, tile_height-v_h%tile_height),(0, tile_width-v_w%tile_width),(0, 0)], mode='constant', constant_values=0)
    # imnpvertpad = imnpvertpad / 255
    # tilesvert=reshape_split(imnpvertpad, (1024,1024))
    # v_h2,v_w2,_=imnpvertpad.shape
    # del imnpvert,imnpvertpad

    imnphv = imnp[512:,512:,:]
    del imnp
    hv_h,hv_w,_=imnphv.shape
    imnphvpad =np.pad(imnphv, pad_width=[(0, tile_height-hv_h%tile_height),(0, tile_width-hv_w%tile_width),(0, 0)], mode='constant', constant_values=0)
    imnphvpad = imnphvpad / 255
    tileshv=reshape_split(imnphvpad, (1024,1024))
    hv_h2,hv_w2,_=imnphvpad.shape
    del imnphv,imnphvpad



    wsipop,wsialone = classifyDL_v2(tiles)
    del tiles
    # wsipop_h,wsialone_h = classifyDL_v2(tileshorz)
    # del tileshorz
    # wsipop_v,wsialone_v = classifyDL_v2(tilesvert)
    # del tilesvert
    wsipop_hv,wsialone_hv = classifyDL_v2(tileshv)
    del tileshv

    wsialone = stitch(wsialone, h, w, h2, w2, channels=1)
    # wsialone_h = stitch(wsialone_h, h_h, h_w, h_h2, h_w2, channels=1)
    # wsialone_v = stitch(wsialone_v, v_h, v_w, v_h2, v_w2, channels=1)
    wsialone_hv = stitch(wsialone_hv, hv_h, hv_w, hv_h2, hv_w2, channels=1)
    wsialone = wsialone[512:, 512:]
    # wsialone_h = wsialone_h[:, 512:]
    # wsialone_v = wsialone_v[512:, :]
    wsipop_t = np.mean(np.stack([wsialone, wsialone_hv]), dtype=np.uint8, axis=0)
    png = Image.fromarray(wsipop_t)
    png = png.convert("L")
    png.save(os.path.join(dst, imnm.replace('.tif', 'single.png')))

    wsipop = stitch(wsipop,h,w,h2,w2,channels=13)
    # wsipop_h = stitch(wsipop_h,h_h,h_w,h_h2,h_w2,channels=13)
    # wsipop_v = stitch(wsipop_v,v_h,v_w,v_h2,v_w2,channels=13)
    wsipop_hv = stitch(wsipop_hv,hv_h,hv_w,hv_h2,hv_w2,channels=13)
    wsipop = wsipop[512:,512:,:]
    # wsipop_h = wsipop_h[:,512:,:]
    # wsipop_v = wsipop_v[512:,:,:]

    # wsipop_t = np.mean(np.stack([wsipop,wsipop_h,wsipop_v,wsipop_hv]),dtype=np.float16,axis=0)
    # wsipop_avg = np.argmax(wsipop_t, axis=2).astype('uint8')
    wsipop_t = np.mean(np.stack([wsipop,wsipop_hv]),dtype=np.float16,axis=0)
    wsipop_avgsmall = np.argmax(wsipop_t, axis=2).astype('uint8')

    # png = Image.fromarray(np.squeeze(wsipop_avg))
    # png = png.convert("L")
    # png.save(os.path.join(dst,imnm.replace('.tif','avg.png')))
    png = Image.fromarray(np.squeeze(wsipop_avgsmall))
    png = png.convert("L")
    png.save(os.path.join(dst,imnm.replace('.tif','avgsm.png')))