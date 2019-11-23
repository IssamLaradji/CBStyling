## Class-Based Styling
### Real-time Localized Style Transfer with Semantic Segmentation
##### ICCV 2019 Computer Vision for Fashion, Art and Design 
[[Paper]]()[[Video]](https://www.youtube.com/watch?v=A_SwsM7Ox5M)

### Running the code
Given an input image, our method CBS can stylize a certain object class with a specif style. Run
the following command to achieve the result below.

```
python main.py -i images/example.jpg -o images/stylized.jpg -s styles/mosaic.pth -c pedestrian
```


Input Image          |  Stylized Cars with CBS
:-------------------------:|:-------------------------:
![original image](results/gt_image.png) |  ![predicted image](results/pred_image.png)

### Choosing the style and the object class
Styles          |  Object Classes
:-------------------------:|:-------------------------:
![original image](results/gt_image.png) |  ![predicted image](results/pred_image.png)


## Citation 
If you find the code useful for your research, please cite:

```
@misc{kurzman2019classbased,
    title={Class-Based Styling: Real-time Localized Style Transfer with Semantic Segmentation},
    author={Lironne Kurzman and David Vazquez and Issam Laradji},
    year={2019},
    eprint={1908.11525},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
