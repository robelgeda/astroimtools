# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tempfile
from numpy.testing import assert_allclose, assert_almost_equal

import os
import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
from astropy.io.fits.tests import FitsTestCase
from astropy.coordinates import SkyCoord

from ..cutout_tools import cutout_tool


class TestCutoutTool(FitsTestCase):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp(prefix='fits-test-')
        fits.conf.enable_record_valued_keyword_cards = True
        fits.conf.extension_name_case_sensitive = False
        fits.conf.strip_header_whitespace = True
        fits.conf.use_memmap = True

    @staticmethod
    def construct_test_image():
        # Construct data: 10 X 10 array where
        # the value at each pixel (x, y) is x*10 + y
        data = np.array([i * 10 + np.arange(10) for i in range(10)])

        # Construct a WCS for test image
        # N.B: Pixel scale should be 1 deg/pix
        w = wcs.WCS()
        w.wcs.crpix = [5, 5]
        w.wcs.crval = [0, 45]
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']

        # Construct fits header
        h = w.to_header()

        # Construct Image and return:
        return fits.PrimaryHDU(data, h)

        def temp(self, filename):
            """ Returns the full path to a file in the test temp dir."""

            return

    def test_cutout_tool_inputs(self):
        # Construct image:
        image_hdu = self.construct_test_image()

        # Construct catalog
        ra = [0, 1] * u.deg
        dec = [45, 46] * u.deg
        ids = ["Target_1", "Target_2"]
        cutout_width = cutout_height = [4.0, 4.0] * u.pix

        catalog = Table(
            data=[ids, ra, dec, cutout_width, cutout_height],
            names=['id', 'ra', 'dec', 'cutout_width', 'cutout_height'])

        # Test with no rotation

        # - From memory:
        assert None not in cutout_tool(image_hdu, catalog)

        # - From fits file:
        fits_file = self.temp("input_image.fits")
        image_hdu.writeto(fits_file)
        assert None not in cutout_tool(fits_file, catalog)

        # - From HDUList:
        hdu_list = fits.HDUList(image_hdu)
        assert None not in cutout_tool(hdu_list, catalog)

        # - From Array and WCS:
        array = image_hdu.data
        w = wcs.WCS(image_hdu.header)
        assert None not in cutout_tool(array, catalog, wcs=w)

        # - From ECSV file
        ecsv_file = self.temp("input_catalog.ecsv")
        catalog.write(ecsv_file, format="ascii.ecsv")
        assert None not in cutout_tool(image_hdu, ecsv_file)

        # Test with rotation column:
        pa = [90, 45] * u.deg
        catalog.add_column(pa, name="cutout_pa")
        assert None not in cutout_tool(image_hdu, catalog)

    def test_cutout_tool_correctness(self):
        # Construct image:
        image_hdu = self.construct_test_image()

        # Construct catalog
        ra = [0] * u.deg  # Center pixel
        dec = [45] * u.deg  # Center pixel
        ids = ["target_1"]
        cutout_width = cutout_height = [3.0] * u.pix # Cutout should be 4 by 4

        catalog = Table(
            data=[ids, ra, dec, cutout_width, cutout_height],
            names=['id', 'ra', 'dec', 'cutout_width', 'cutout_height'])

        cutout = cutout_tool(image_hdu, catalog, to_fits=True)[0]

        # check if values are correct:
        w_orig = wcs.WCS(image_hdu.header)
        w_new = wcs.WCS(cutout.header)

        for x_new, x_orig in enumerate(range(3, 6)):
            for y_new, y_orig in enumerate(range(3, 6)):
                coords_orig = SkyCoord.from_pixel(x_orig, y_orig, w_orig, origin=0)
                coords_new = SkyCoord.from_pixel(x_new, y_new, w_new, origin=0)

                assert_almost_equal(image_hdu.data[x_orig][y_orig], cutout.data[x_new][y_new])
                assert_almost_equal(coords_orig.ra.value, coords_new.ra.value)
                assert_almost_equal(coords_orig.dec.value, coords_new.dec.value)

        # Test for rotation:
        pa = [90] * u.deg
        catalog.add_column(pa, name="cutout_pa")

        cutout = cutout_tool(image_hdu, catalog, to_fits=True)[0]

        # check if values are correct:
        w_orig = wcs.WCS(image_hdu.header)
        w_new = wcs.WCS(cutout.header)

        for x_new, x_orig in enumerate(range(6, 3, -1)):
            for y_new, y_orig in enumerate(range(6, 3, -1)):
                coords_orig = SkyCoord.from_pixel(x_orig, y_orig, w_orig, origin=0)
                coords_new = SkyCoord.from_pixel(x_new, y_new, w_new, origin=0)

                assert_almost_equal(image_hdu.data[x_orig][y_orig], cutout.data[x_new][y_new])
                assert_almost_equal(coords_orig.ra.value, coords_new.ra.value)
                assert_almost_equal(coords_orig.dec.value, coords_new.dec.value)
