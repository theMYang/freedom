import os
from datetime import datetime

from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Add, Subtract
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
# from keras.applications import VGG16
from keras import initializers
from keras import regularizers
from keras import backend as K
from libs.pconv_layer import PConv2D


class PConvUnet(object):

    def __init__(self, img_rows=32, img_cols=32, channels=1, weight_filepath=None):
        
        # 配置
        self.weight_filepath = weight_filepath
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        assert self.img_rows >= 3, 'Height must be >3 pixels'
        assert self.img_cols >= 3, 'Width must be >3 pixels'

        # 当前epoch
        self.current_epoch = 0
        
        # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)未使用
        self.vgg_layers = [3, 6, 10]
        
        # Get the vgg16 model for perceptual loss        
#         self.vgg = self.build_vgg()
        
        # Create UNet-like model
        self.model = self.build_pconv_unet()
        
    def build_vgg(self):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """
        # Input image to extract features from
        img = Input(shape=(self.img_rows, self.img_cols, self.channels))

        # Get the vgg network from Keras applications
        vgg = VGG16(weights="imagenet", include_top=False)

        # Output the first three pooling layers
        vgg.outputs = [vgg.layers[i].output for i in self.vgg_layers]

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')
        
        return model
        
    def build_pconv_unet(self, train_bn=True, lr=0.0002):      

        # INPUTS
        inputs_img = Input((self.img_rows, self.img_cols, self.channels))
        inputs_mask = Input((self.img_rows, self.img_cols, self.channels))
        # kernel_init = initializers.he_normal()
        # bias_init = initializers.he_normal()
        kernel_init = 'glorot_uniform'
        bias_init = 'zeros'

#         kernel_init = initializers.he_uniform()
#         bias_init = 'Orthogonal'
        kernel_regul = regularizers.l2(1)
        activity_regul = regularizers.l2(1)
        


        # ResNet block
        def identity_block(X, filters, f):

            F1, F2 = filters

            X_shortcut = X

            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            X = Conv2D(filters=F1, kernel_size=(f, f), strides=(1, 1), padding='same',
                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                      kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(X)

            X = BatchNormalization(axis=3)(X)
            X = Activation('relu')(X)
            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                       kernel_initializer=kernel_init, bias_initializer=bias_init,
                      kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(X)
            
            X = Add()([X, X_shortcut])
            X = Activation('relu')(X)

            return X

        # ENCODER
        def encoder_layer(img_in, filters, kernel_size, bn=True, resid=True):
            # conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(img_in)
            conv = img_in
            if bn:
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
#             conv = MaxPooling2D((2, 2))(conv)
            

            if resid:
                conv = identity_block(conv, (filters, filters), kernel_size)
            encoder_layer.counter += 1
            return conv

        # DECODER
        def decoder_layer(img_in, e_conv, filters, kernel_size, bn=True, resid=True):
            # up_img = UpSampling2D(size=(2,2))(img_in)
            up_img = img_in
            concat_img = Concatenate(axis=3)([e_conv,up_img])
            conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same',
                          kernel_initializer=kernel_init, bias_initializer=bias_init,
                      kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(concat_img)
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0)(conv)

            if resid:
                conv = identity_block(conv, (filters, filters), kernel_size)
            return conv

        encoder_layer.counter = 0
        filters_base = 32
        e_conv1_head = Conv2D(filters=filters_base, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=kernel_init, bias_initializer=bias_init,
                      kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(inputs_img)
#         e_conv1_head = Conv2D(filters=filters_base*1, kernel_size=3, strides=1, padding='same',
#                               kernel_initializer=kernel_init, bias_initializer=bias_init,
#                       kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(e_conv1_head)
        e_conv1_tail = AveragePooling2D((2, 2))(e_conv1_head)
        e_conv1 = encoder_layer(e_conv1_tail, filters_base, 3, bn=False)

        e_conv2_head = Conv2D(filters=filters_base*2, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=kernel_init, bias_initializer=bias_init,
                      kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(e_conv1)
        e_conv2_tail = AveragePooling2D((2, 2))(e_conv2_head)
        e_conv2 = encoder_layer(e_conv2_tail, filters_base*2, 3)

        e_conv3_head = Conv2D(filters=filters_base*4, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=kernel_init, bias_initializer=bias_init,
                      kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(e_conv2)
        e_conv3_tail = AveragePooling2D((2, 2))(e_conv3_head)
        d_conv3_head = encoder_layer(e_conv3_tail, filters_base*4, 3)
        resid1 = Subtract()([e_conv3_tail, d_conv3_head])
        d_conv3_tail = UpSampling2D(size=(2, 2))(resid1)
        
        
        d_conv4_head = decoder_layer(d_conv3_tail, e_conv3_head, filters_base*2, 3)
        resid2 = Subtract()([d_conv4_head, e_conv2_tail])
        d_conv4_tail = UpSampling2D(size=(2, 2))(resid2)
        

        d_conv5_head = decoder_layer(d_conv4_tail, e_conv2_head, filters_base*1, 3)
        resid3 = Subtract()([d_conv5_head, e_conv1_tail])
        d_conv5_tail = UpSampling2D(size=(2, 2))(resid3)
        

        d_conv6_head = decoder_layer(d_conv5_tail, e_conv1_head, filters_base//2, 3, bn=False)

#         outputs = Conv2D(1, 1, activation = 'relu',
        outputs = Conv2D(1, 1, activation = 'relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
                      kernel_regularizer=kernel_regul, bias_regularizer=activity_regul)(d_conv6_head)

        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)
        inputs_mask = inputs_mask[:, :, :, :1]

        # Compile the model
        model.compile(
            optimizer = Adam(lr=0.004),
            loss=self.loss_total(inputs_mask)
        )

        return model
    
    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components 
        and multiplies by their weights. See paper eq. 7.
        """
        def loss(y_true, y_pred):

            # Compute predicted image with non-hole pixels set to ground truth
            # y_comp = mask * y_true + (1-mask) * y_pred

            # Compute the vgg features
            # vgg_out = self.vgg(y_pred)
            # vgg_gt = self.vgg(y_true)
            # vgg_comp = self.vgg(y_comp)
            
            # Compute loss components
#             l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            # l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp)
            # l4 = self.loss_style(vgg_out, vgg_gt)
            # l5 = self.loss_style(vgg_comp, vgg_gt)
            # l6 = self.loss_tv(mask, y_comp)
            
            # Return loss function
            return l2
#             return l1 + 6*l2
            # return l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6

        return loss
    
    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l2((1-mask) * y_true, (1-mask) * y_pred)
    
    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l2(mask * y_true, mask * y_pred)
    
    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
        """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss
        
    def loss_style(self, output, vgg_gt):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
        return loss
    
    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])        
        return a+b

    def get_optimizer(self):
        return self.model.optimizer

    def fit(self, generator, epochs=3, plot_callback=None, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator
        
        param generator: training generator yielding (maskes_image, original_image) tuples
        param epochs: number of epochs to train for
        param plot_callback: callback function taking Unet model as parameter
        """
        
        # Loop over epochs
        for _ in range(epochs):            
            
            # Fit the model
            self.model.fit_generator(
                generator,
                epochs=self.current_epoch+1,
                initial_epoch=self.current_epoch,
                *args, **kwargs
            )

            # Update epoch 
            self.current_epoch += 1
            
            # After each epoch predict on test images & show them
            if plot_callback:
                plot_callback(self.model)

            # Save logfile
            if self.weight_filepath:
                self.save()
            
    def predict(self, sample, batch_size=32):
        """Run prediction using this model"""
        return self.model.predict(sample, batch_size=batch_size)

    def predict_generator(self, generator, steps=None):
        """Run prediction using this model"""
        return self.model.predict_generator(generator, steps)

    def evaluate_generator(self, generator, steps=None):
        """Run prediction using this model"""
        return self.model.evaluate_generator(generator, steps)

    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def save(self, path):        
        #self.model.save_weights(self.current_weightfile())
        self.model.save(path)
        
    def load_weights(self, path): 
        self.model.load_weights(path)
        
    def save_weights(self, path): 
        self.model.save_weights(path)

    def load(self, filepath, train_bn=True, lr=0.0002):

        # Create UNet-like model
        self.model = self.build_pconv_unet(train_bn, lr)

        # Load weights into model
        epoch = int(os.path.basename(filepath).split("_")[0])
        assert epoch > 0, "Could not parse weight file. Should start with 'X_', with X being the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)        

    def current_weightfile(self):
        assert self.weight_filepath != None, 'Must specify location of logs'
        return self.weight_filepath + "{}_weights_{}.h5".format(self.current_epoch, self.current_timestamp())

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
            
    @staticmethod
    def l2(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.sum(K.square(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.sum(K.square(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
    
    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""
        
        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
        
        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        
        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        
        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)
        
        return gram
