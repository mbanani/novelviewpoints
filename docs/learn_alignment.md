# Learn Alignment 

We learn alignment by training the model found in `models/model_3d_realism.py`. This model is
trained on image pairs.

## Model Training
Coming soon. 

## Model inference

You can run inference by running the command below: 

```
python evaluate.py --checkpoint <model_checkpoint> --n_views 2 --dataset <dataset> --split test
```

We also release the pretrained weights use to obtain the results in the paper, which can be
obtained by running the following script. 

```bash
# Download pretrained weights
wget http://www-personal.umich.edu/~mbanani/novelviewpoints/realism_pix3d_depth.pkl
wget http://www-personal.umich.edu/~mbanani/novelviewpoints/realism_pix3d_mask.pkl
wget http://www-personal.umich.edu/~mbanani/novelviewpoints/realism_shapenet_depth.pkl
wget http://www-personal.umich.edu/~mbanani/novelviewpoints/realism_shapenet_mask.pkl
wget http://www-personal.umich.edu/~mbanani/novelviewpoints/realism_thingi10k_depth.pkl
wget http://www-personal.umich.edu/~mbanani/novelviewpoints/realism_thingi10k_mask.pkl
```
