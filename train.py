# create our checkopint manager
ckpt = tf.train.Checkpoint(net=net)
ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=2)

# load from a previous checkpoint if it exists, else initialize the model from scratch

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  start_interation = int(ckpt_manager.latest_checkpoint.split("-")[-1])
  print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
  print("Initializing from scratch.")

loss_fn = tf.keras.losses.MeanSquaredError()
# Prepare the metrics.
train_acc_metric = tf.keras.metrics.MeanSquaredError('mse train')
val_acc_metric = tf.keras.metrics.MeanSquaredError('mse val')
# Optimizers
opt = tfa.optimizers.AdamW(
      learning_rate=learning_rate, weight_decay=weight_decay)


@tf.function
def train_step(batch):
    timestep_values = generate_timestamp(batch.shape[0])
    noised_image, noise = forward_noise(batch, timestep_values)
    with tf.GradientTape() as tape:
        prediction = net(noised_image, timestep_values)

        loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, net.trainable_variables)
    opt.apply_gradients(zip(gradients, net.trainable_variables))
    train_acc_metric.update_state(noise, prediction)
    return loss_value


@tf.function
def test_step(batch):
    timestep_values = generate_timestamp(batch.shape[0])

    noised_image, noise = forward_noise(batch, timestep_values)

    prediction = net(noised_image, timestep_values)
    loss_value = loss_fn(noise, prediction)
    # Update training metric.
    val_acc_metric.update_state(noise, prediction)
    return loss_value


from tqdm.notebook import trange
import time

for e in trange(num_epochs):
    print("\nStart of epoch %d" % (e,))
    start_time = time.time()

    # this is cool utility in Tensorflow that will create a nice looking progress bar
    for i, batch in enumerate(iter(train_generator)):
        # run the training loop
        loss = train_step(batch)

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()

    print("Training MSE: %.4f" % (float(train_acc),))
    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    for i, batch in enumerate(iter(val_generator)):
        # run the training loop
        val_loss = test_step(batch)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()

    print("Validation MSE: %.4f" % (float(val_acc),))
    # print("validation KID: %.4f" % (float(val_kid),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    ckpt_manager.save(checkpoint_number=e)
