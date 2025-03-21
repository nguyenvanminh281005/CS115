class Forward_Noise(tf.keras.layers.Layer):


def __init__(self, sqrt_alpha_bar, one_minus_sqrt_alpha_bar, **kwargs):
    super(Forward_Noise, self).__init__(**kwargs)
    self.sqrt_alpha_bar = sqrt_alpha_bar
    self.one_minus_sqrt_alpha_bar = one_minus_sqrt_alpha_bar


def forward_noise(self, x_0, t):
    noise = tf.random.normal(x_0.shape)
    reshaped_sqrt_alpha_bar_t = tf.cast(
        tf.experimental.numpy.reshape(tf.experimental.numpy.take(self.sqrt_alpha_bar, t), (-1, 1, 1, 1)),
        tf.float32)  # trung bình tổng thể
    reshaped_one_minus_sqrt_alpha_bar_t = tf.cast(
        tf.experimental.numpy.reshape(tf.experimental.numpy.take(self.one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1)),
        tf.float32)  # phương sai
    noisy_image = reshaped_sqrt_alpha_bar_t * x_0 + reshaped_one_minus_sqrt_alpha_bar_t * noise  # Reparameter trick
    return noisy_image, noise


def call(self, x_0, t):
    noise_img, noise = self.forward_noise(x_0, t)
    return noise_img, noise


class Generator(tf.keras.layers.Layer):
    def __init__(self, timesteps, **kwargs):
        super(Generator, self).__init__()
        self.timesteps = timesteps


def generate_timestamp(self, num):
    return tf.random.uniform(shape=[num], minval=0, maxval=self.timesteps, dtype=tf.int32)


def call(self, x):
    x = self.generate_timestamp(x)
    return x

forward_noise = Forward_Noise(sqrt_alpha_bar,one_minus_sqrt_alpha_bar)
generate_timestamp = Generator(timesteps)
train_generator = DataGenerator(x_train, image_size = image_size,batch_size =batch_size,shuffle=True)
val_generator = DataGenerator(x_test,image_size = image_size,batch_size=batch_size,shuffle=True)
# Let us visualize the output image at a few timestamps
sample_mnist = train_generator.__getitem__(1)[0]
fig = plt.figure(figsize=(32,50))

for index, i in enumerate([0,50,100,150,200,250,300,350,400,450,500]):
  noisy_im, noise = forward_noise(sample_mnist, np.array([i,]))
  plt.subplot(1, 11, index+1)
  plt.imshow(np.squeeze(noisy_im))
plt.show()

