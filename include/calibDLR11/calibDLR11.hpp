/*
    IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

    By downloading, copying, installing or using the software you agree
    to this license. If you do not agree to this license, do not
    download, install, copy or use the software.

                             License Agreement
                  For Open Source Computer Vision Library

    Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
    Copyright (C) 2009, Willow Garage Inc., all rights reserved.
    Copyright (C) 2018, Wenfeng CAI, all rights reserved.
    Third party copyrights are property of their respective owners.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

      * Redistribution's of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

      * Redistribution's in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.

      * The name of the copyright holders may not be used to endorse or
        promote products derived from this software without specific
        prior written permission.

    This software is provided by the copyright holders and contributors
    "as is" and any express or implied warranties, including, but not
    limited to, the implied warranties of merchantability and fitness
    for a particular purpose are disclaimed.  In no event shall the
    Intel Corporation or contributors be liable for any direct,
    indirect, incidental, special, exemplary, or consequential damages
    (including, but not limited to, procurement of substitute goods or
    services; loss of use, data, or profits; or business interruption)
    however caused and on any theory of liability, whether in contract,
    strict liability, or tort (including negligence or otherwise)
    arising in any way out of the use of this software, even if advised
    of the possibility of such damage.
 */

#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/imgproc/detail/distortion_model.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <iterator>
#include <stdio.h>

namespace dlr11 {

using namespace cv;

/*!
    \brief Finds the camera intrinsic and extrinsic parameters from
    several views of a calibration pattern.

    \param imagePoints It is a vector of vectors of the projections of
    calibration pattern points (e.g.
    std::vector<std::vector<cv::Vec2f>>). imagePoints[i].size() must be
    equal to objectPoints.size() for each i.

    \param imageSize Size of the image used only to initialize the
    intrinsic camera matrix.

    \param objectPoints It is a vector of calibration pattern points in
    the calibration pattern coordinate space (e.g.
    std::vector<cv::Vec3f>). The same calibration pattern must be shown
    in each view and it is fully visible. The points are 3D, but since
    they are in a pattern coordinate system, then, if the rig is planar,
    it may make sense to put the model to a XY coordinate plane so that
    Z-coordinate of each input object point is 0. On output, the refined
    pattern points are returned for imperfect planar target.

    \param cameraMatrix Output 3x3 floating-point camera matrix \f$A =
    \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If
    CV_CALIB_USE_INTRINSIC_GUESS and/or CALIB_FIX_ASPECT_RATIO are
    specified, some or all of fx, fy, cx, cy must be initialized before
    calling the function.

    \param distCoeffs Output vector of distortion coefficients \f$(k_1,
    k_2, p_1, p_2[, k_3[, k_4, k_5, k_6 [, s_1, s_2, s_3, s_4[, \tau_x,
    \tau_y]]]])\f$ of 4, 5, 8, 12 or 14 elements.

    \param rvecs Output vector of rotation vectors (see Rodrigues )
    estimated for each pattern view (e.g. std::vector<cv::Mat>>). That
    is, each k-th rotation vector together with the corresponding k-th
    translation vector (see the next output parameter description)
    brings the calibration pattern from the model coordinate space (in
    which object points are specified) to the world coordinate space,
    that is, a real position of the calibration pattern in the k-th
    pattern view (k=0.. *M* -1).

    \aram tvecs Output vector of translation vectors estimated for each
    pattern view.

    \param newObjPoints The updated output vector of pattern points.

    \param stdDeviationsIntrinsics Output vector of standard deviations
    estimated for intrinsic parameters. Order of deviations values:
    \f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1,
    s_2, s_3, s_4, \tau_x, \tau_y)\f$ If one of parameters is not
    estimated, it's deviation is equals to zero.

    \param stdDeviationsExtrinsics Output vector of standard deviations
    estimated for extrinsic parameters. Order of deviations values:
    \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern
    views, \f$R_i, T_i\f$ are concatenated 1x3 vectors.

    \param perViewErrors Output vector of the RMS re-projection error
    estimated for each pattern view.

    \param flags Different flags that may be zero or a combination of
    the following values:
        -   **CALIB_USE_INTRINSIC_GUESS** cameraMatrix contains valid
            initial values of fx, fy, cx, cy that are optimized further.
            Otherwise, (cx, cy) is initially set to the image center (
            imageSize is used), and focal distances are computed in a
            least-squares fashion. Note, that if intrinsic parameters
            are known, there is no need to use this function just to
            estimate extrinsic parameters. Use solvePnP instead.
        -   **CALIB_FIX_PRINCIPAL_POINT** The principal point is not
            changed during the global optimization. It stays at the
            center or at a different location specified when
            CALIB_USE_INTRINSIC_GUESS is set too.
        -   **CALIB_FIX_ASPECT_RATIO** The functions considers only fy
            as a free parameter. The ratio fx/fy stays the same as in
            the input cameraMatrix. When CALIB_USE_INTRINSIC_GUESS is
            not set, the actual input values of fx and fy are ignored,
            only their ratio is computed and used further.
        -   **CALIB_ZERO_TANGENT_DIST** Tangential distortion coefficients
            \f$(p_1, p_2)\f$ are set to zeros and stay zero.
        -   **CALIB_FIX_K1,...,CALIB_FIX_K6** The corresponding radial
            distortion coefficient is not changed during the
            optimization. If CALIB_USE_INTRINSIC_GUESS is set, the
            coefficient from the supplied distCoeffs matrix is used.
            Otherwise, it is set to 0.
        -   **CALIB_RATIONAL_MODEL** Coefficients k4, k5, and k6 are
            enabled. To provide the backward compatibility, this extra
            flag should be explicitly specified to make the calibration
            function use the rational model and return 8 coefficients.
            If the flag is not set, the function computes and returns
            only 5 distortion coefficients.
        -   **CALIB_THIN_PRISM_MODEL** Coefficients s1, s2, s3 and s4
            are enabled. To provide the backward compatibility, this
            extra flag should be explicitly specified to make the
            calibration function use the thin prism model and return 12
            coefficients. If the flag is not set, the function computes
            and returns only 5 distortion coefficients.
        -   **CALIB_FIX_S1_S2_S3_S4** The thin prism distortion
            coefficients are not changed during the optimization. If
            CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
            supplied distCoeffs matrix is used. Otherwise, it is set to 0.
        -   **CALIB_TILTED_MODEL** Coefficients tauX and tauY are
            enabled. To provide the backward compatibility, this extra
            flag should be explicitly specified to make the calibration
            function use the tilted sensor model and return 14
            coefficients. If the flag is not set, the function computes
            and returns only 5 distortion coefficients.
        -   **CALIB_FIX_TAUX_TAUY** The coefficients of the tilted
            sensor model are not changed during the optimization. If
            CALIB_USE_INTRINSIC_GUESS is set, the coefficient from the
            supplied distCoeffs matrix is used. Otherwise, it is set to 0.

    \param criteria Termination criteria for the iterative optimization
    algorithm.

    \return the overall RMS re-projection error.

    The function estimates the intrinsic camera parameters and extrinsic
    parameters for each of the views. The algorithm is based on Klaus H.
    Strobl and Gerd Hirzinger's paper "More Accurate Pinhole Camera
    Calibration with Imperfect Planar Target". The coordinates of 3D
    object points and their corresponding 2D projections in each view
    must be specified. That may be achieved by using an object with a
    known geometry and easily detectable feature points. Such an object
    is called a calibration rig or calibration pattern. Currently,
    initialization of intrinsic parameters (when CALIB_USE_INTRINSIC_GUESS
    is not set) is only implemented for planar calibration patterns
    (where Z-coordinates of the object points must be all zeros).

    The algorithm performs the following steps:

    -   Compute the initial intrinsic parameters (the option only
        available for planar calibration patterns) or read them from the
        input parameters. The distortion coefficients are all set to
        zeros initially unless some of CALIB_FIX_K? are specified.

    -   Estimate the initial camera pose as if the intrinsic parameters
        have been already known. This is done using solvePnP.

    -   Run the global Levenberg-Marquardt optimization algorithm to
        minimize the reprojection error, that is, the total sum of
        squared distances between the observed feature points
        imagePoints and the projected (using the current estimates for
        camera parameters and the poses) object points objectPoints. See
        projectPoints for details.
 */
double calibrateCamera(InputArrayOfArrays _imagePoints, Size imageSize,
    InputArray _objectPoints, int _fixedObjPt, InputOutputArray _cameraMatrix,
    InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
    OutputArrayOfArrays _tvecs, OutputArray _newObjPoints,
    OutputArray stdDeviationsIntrinsics, OutputArray stdDeviationsExtrinsics,
    OutputArray _perViewErrors, int flags = 0,
    TermCriteria criteria = TermCriteria(
        TermCriteria::COUNT + TermCriteria::EPS, 60, DBL_EPSILON * 30));

/*!
    \overload double calibrateCamera(InputArrayOfArrays imagePoints,
        Size imageSize, InputArray objectPoints, int _fixedObjPt,
        InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
        OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
        OutputArray newObjPoints, OutputArray stdDeviations, OutputArray
        perViewErrors, int flags = 0, TermCriteria criteria =
        TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 60,
        DBL_EPSILON * 30))
 */
double calibrateCamera(InputArrayOfArrays _imagePoints, Size imageSize,
    InputArray _objectPoints, int _fixedObjPt, InputOutputArray _cameraMatrix,
    InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs,
    OutputArrayOfArrays _tvecs, OutputArray _newObjPoints, int flags = 0,
    TermCriteria criteria = TermCriteria(
        TermCriteria::COUNT + TermCriteria::EPS, 60, DBL_EPSILON * 30));

} /* end of namespace dlr11 */
