# Datasets 

We use three datasets in this work: 
[ShapeNet version 2](https://www.shapenet.org), 
[Thingi10k](https://ten-thousand-models.appspot.com),
and [Pix3D](http://pix3d.csail.mit.edu). 
We make use of the 3D models provided by each of those datasets, and we adapt the rendering
code from 
[RenderForCNN](https://github.com/shapenet/RenderForCNN/) 
to render the models. 

We first generate a `render_dict` that has N random viewpoints for each model in the dataset. 
Then, we use blender to obtain the RGB-A and Depth for each of those viewpoints.
Finally, we overlay the rendered models on random background images for the
[SUN dataset](https://groups.csail.mit.edu/vision/SUN/) as done in RenderForCNN. 
 
For your convenience, we make our dataset dictionaries and rendered images available. 
Please use the following script to download the datasets. Make sure to adjust the data paths in
`datasets/dataset_generator.py`

```bash
# Download rendered images
Coming soon.

```


## Dataset download and preprocessing

Due to copyright issues, we cannot directly provide the download links to those datasets.
However, the datasets are all freely available through their respective websites. 
Once the datasets are downloaded, you can use the following script to generate the `render_dict`
for each dataset, use it to render the models, and overlay the backgrounds.

```bash
Coming Soon.
```

## Rendering 

We use Blender 2.79 to render the images, and adapt the pipeline used in Render For CNN for those purposes. 
The rendering process takes as input a dictionary for model IDs and camera poses and outputs RGBA png images and EXR depth images.
The rendering code will be made available soon.

As noted in the paper, we normalize the scale of Thingi10k models to a unit-square. This has
already been done to other datasets; tables, planes, and chairs all have the same spatial
extent in ShapeNet and Pix3D. 
