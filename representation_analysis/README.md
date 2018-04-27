# Representation Analysis Project

You can train the VAE one of two ways:

## From Google Drive

1) To train the VAE from images, download the training data from the following link:

```https://drive.google.com/open?id=1QujAUmmbfv2NNiyojrpjDur8FfC4xhgN```

Make sure the data is in the following directory:

``` gym_duckietown/representation_analysis/```

Do not remove the directory structure of the `data/` folder! Get a coffee, because downloading and unzipping 
will take a while

2) From `gym_duckietown` run the following command:

``` python representation_analysis/train_vae.py ```

## Generating Environment on the fly

1) From `gym_duckietown` run the following command:

``` python  representation_analysis/depricated/train_vae.py```

## Evaluating Disentanglement

1) Collect data

``` python get_test_data.py ```

2) Compute Higgens metric

``` python evaluate_disentanglement.py --saved_model some_checkpoint.ckpt ```

3) ???

4) Profit
