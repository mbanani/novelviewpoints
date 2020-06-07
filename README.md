Novel Object Viewpoint Estimation through Reconstruction Alignment
====================================

Code for our CVPR 2020 paper:

**[Novel Object Viewpoint Estimation through Reconstruction Alignment][1]**  
Mohamed El Banani, Jason Corso, David Fouhey

If you find this code useful, please consider citing:  
```text
@inProceedings{elbanani2020novelviewpoints,
  title={Novel Object Viewpoint Estimation through Reconstruction Alignment},
  author={{El Banani}, Mohamed and Corso, Jason J. and Fouhey, David},
  booktitle={Computer Vision and Pattern Recognition (CVPR)}
  year={2020},
}
```

If you have any questions about the paper or the code, please feel free to email me at
mbanani@umich.edu 


Usage Instructions
------------------

1. [How to setup your environment?][2]
2. [How to download and render the 3D models?][3]
4. [How to train your network to learn shape?][4]
4. [How to train your network to learn alignment?][5]
5. [How to run relative viewpoint inference?][6]


Acknowledgments
---------------
We would like to thank the reviewers and area chairs for their valuable comments and suggestions,
and the members of the UM AI Lab for many helpful discussions. 
Toyota Research Institute ("TRI") provided funds to assist the authors with their research but this
article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota
entity.

We would also like to acknowledge the following repositories and users for making great code openly available for us to use:

- [@pytorch/pytorch](https://www.github.com/pytorch/pytorch) because I can't imagine doing this
  in CAFFE.
- [@akar43/lsm](https://github.com/akar43/lsm) for providing very readable projection code. 
- [@ShapeNet/RenderForCNN](https://github.com/shapenet/RenderForCNN) for making their rendering
  code openly available. 
- [Karan Desai](https://github.com/kdexd) and [Richard Higgins](https://github.com/relh) for being great resources on how to write better code. 


[1]: https://mbanani.github.io/novelviewpoints/
[2]: https://github.com/mbanani/novelviewpoints/tree/master/docs/setup.md 
[3]: https://github.com/mbanani/novelviewpoints/tree/master/docs/datasets.md 
[4]: https://github.com/mbanani/novelviewpoints/tree/master/docs/learn_shape.md 
[5]: https://github.com/mbanani/novelviewpoints/tree/master/docs/learn_alignment.md 
[6]: https://github.com/mbanani/novelviewpoints/tree/master/docs/infer_viewpoint.md 
