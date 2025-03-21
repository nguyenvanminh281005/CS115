def ddpm(x_t, pred_noise, t):
   alpha_t = np.take(alpha, t)
   alpha_t_bar = np.take(alpha_bar, t)

   eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
   mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

   var = np.take(beta, t)
   z = np.random.normal(size=x_t.shape)

   return mean + (var ** .5) * z


from PIL import Image


def save_gif(img_list, path="", interval=500):
    # Transform images from [-1,1] to [0, 255]
    imgs = []
    for im in img_list:
        im = np.array(im)
        im = (im + 1) * 127.5
        im = np.clip(im, 0, 255).astype(np.int32)
        im = Image.fromarray(im)
        imgs.append(im)

    imgs = iter(imgs)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(fp=path, format='GIF', append_images=imgs,
             save_all=True, duration=interval, loop=0)


from tqdm import tqdm

x = tf.random.normal((1, 32, 32, 1))
img_list = []
img_list.append(np.squeeze(np.squeeze(x, 0), -1))
for i in tqdm(range(timesteps)):
    t = np.expand_dims(np.array(timesteps - i, np.int32), 0)
    pred_noise = net(x, t)
    x = ddpm(x, pred_noise, t)

    img_list.append(np.squeeze(np.squeeze(x, 0), -1))
    if i % 25 == 0:
        img = np.squeeze(x[0])
        plt.imshow(np.array(np.clip((img + 1) * 127.5, 0, 255), np.uint8))
        plt.show()
save_gif(img_list + ([img_list[-1]] * 100), "ddpm.gif", interval=20)
plt.imshow(np.array(np.clip(img, a_min=0, a_max=255)))
plt.show()
