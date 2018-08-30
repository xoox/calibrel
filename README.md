### calibDLR11: More Accurate Camera Calibration with Imperfect Planar Target

This calibration code is based on the paper:
K. H. Strobl and G. Hirzinger. "More Accurate Pinhole Camera Calibration
with Imperfect Planar Target." In Proceedings of the IEEE International
Conference on Computer Vision (ICCV 2011), 1st IEEE Workshop on
Challenges and Opportunities in Robot Perception, Barcelona, Spain, pp.
1068-1075, November 2011.

The code is largely copied from OpenCV's implementation.

**Note**: the current results are not as good as expected. See
https://github.com/xoox/calibDLR11/issues/1 for test results and further
discussion.

#### How to build

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

#### Run test program

```
test_calibDLR11 --mode=0 default.xml
```

Where `mode` has one of the following three values:

* **0** Test with this DLR11 method.
* **1** Test with OpenCV's calibration method.
* **2** Test with the hybrid method, i.e., OpenCV's calibration method
  followed by DLR11 method.

See [calibDLR11_testdata](https://github.com/xoox/calibDLR11_testdata)
for examples of XML setting files.
