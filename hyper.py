# HYPARAMETER

# data
num_epochs = 50  # train for at least 50 epochs for good results
image_size = 32

# optimization
batch_size = 256
learning_rate = 1e-3
weight_decay = 1e-4

# model
timesteps = 500 # Số bước thời gian T bạn có thể đặt tùy ý như DDPM trong bài báo là 1000 nhưng đây minh đặt theo cảm hứng
a_min = 0.0001 #
a_max = 0.02   #

# create a fixed beta schedule
beta = np.linspace(a_min,a_max, timesteps+1) lịch trình phương sai như trong series 2

# this will be used as discussed in the reparameterization trick
alpha = 1 - beta # Đặt alpha = 1-beta
alpha_bar = np.cumprod(alpha, 0) # Tính toán tích tất cả các alpha
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0) #a_0 = 1
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
