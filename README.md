## Seam Carving

Middlebury College CS453 (Computer Vision) Final Project

Dana Silver and Shannon Ovitt

[Final Report](http://www.cs.middlebury.edu/~dsilver/vision/seam-carving/)

### Install OpenCV

On OS X:

```sh
$ brew install opencv
```

* [Linux](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)
* [Windows](http://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html)

### Usage

Non-interactive usage:

```sh
$ python seamcarve.py image width height
```

* `image` the image to resize
* `width` the new width for the image, smaller than the image's width
* `height` the new height for the image, smaller than the image's height

Interactive usage:

```sh
$ python seamcarve.py image --interactive
```

Both options will open a new window that will animate the seams being removed.
Press any key when the animation is done to close the window.

See the
[final report](http://www.cs.middlebury.edu/~dsilver/vision/seam-carving/)
for full implementation details and results.

### References

* [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/imret.pdf) (PDF)
* [Seam Carving for Content-Aware Image Resizing](http://www.faculty.idc.ac.il/arik/SCWeb/imret/) (Project Page)
* [Seam carving, Wikipedia](https://en.wikipedia.org/wiki/Seam_carving)
