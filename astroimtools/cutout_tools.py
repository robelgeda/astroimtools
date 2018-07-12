# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions for cutout images."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# STDLIB
import os
from functools import partial
import math
import errno

# THIRD-PARTY
import numpy as np

# ASTROPY
import astropy.units as u
from astropy import log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, ImageHDU, CompImageHDU
from astropy.nddata.utils import (Cutout2D, NoOverlapError)
from astropy.nddata.nddata import NDData
from astropy.table import QTable, Table
from astropy.wcs import WCS, NoConvergence
from astropy.wcs.utils import proj_plane_pixel_scales

__all__ = ['make_cutouts', 'show_cutout_with_slit', 'cutout_tool']


def cutout_tool(image, catalog, wcs=None, image_ext=0, origin=0,
                to_fits=False, output_dir=None, overwrite=False,
                delimiter=None, suppress_rotation=False, verbose=True):
    """Make cutouts from a 2D image and write them to FITS files.

    The input Catalog must have the following columns, which must have
    `~astropy.unit.Unit`s where applicable:

        * ``'id'`` - ID string; no unit necessary.
        * ``'ra'`` or ``'x'``- RA (angular units e.g., deg, H:M:S, arcsec etc..)
          or pixel x position (only in `~astropy.units.pix`).
        * ``'dec'`` or ``'y'`` - Dec (angular units e.g., deg, D:M:S, arcsec etc..)
          or pixel y position (only in `~astropy.units.pix`).
        * ``'cutout_width'`` - Cutout width (e.g., in arcsec, pix).
        * ``'cutout_height'`` - Cutout height (e.g., in arcsec, pix).

    Optional columns:
        * ``'cutout_pa'`` - Cutout angle (e.g., in deg, arcsec). This is only
          use if user chooses to rotate the cutouts. Positive value
          will result in a clockwise rotation.

    If saved to fits, cutouts are organized as follows:
        <output_dir>/
            <id>.fits

    Each cutout image is a simple single-extension FITS with updated WCS.
    Its header has the following special keywords:

        * ``OBJ_RA`` - RA of the cutout object in degrees.
        * ``OBJ_DEC`` - DEC of the cutout object in degrees.
        * ``OBJ_ROT`` - Rotation of cutout object in degrees.

    Examples
    --------
    Given a list of Hubble Ultra Deep Field RA and Dec coords,
    you may use the tool as follows:
        >>> from astropy.table import Table
        >>> import astropy.units as u

        >>> ra = [53.18782913, 53.14794797, 53.15059559] * u.deg
        >>> dec = [-27.79405589, -27.77392421, -27.77158621] * u.deg
        >>> ids = ["Galax_0", 123, 53.15059559 * u.deg]
        >>> cutout_width = cutout_height = [3.0, 4.0, 3.0] * u.arcsec

        >>> catalog = Table(
        ...     data=[ids, ra, dec, cutout_width, cutout_height],
        ...     names=['id', 'ra', 'dec', 'cutout_width', 'cutout_height'])

        # To get a list of NDData objects:
        >>> cutouts = cutout_tool('h_udf_wfc_b_drz_img.fits', catalog)
        # To get a list of PrimaryHDU objects:
        >>> cutouts = cutout_tool('h_udf_wfc_b_drz_img.fits', catalog, to_fits=True)
        # To save to fits file provide an output dir:
        >>> cutouts = cutout_tool('h_udf_wfc_b_drz_img.fits', catalog, output_dir='~/cutouts')

        # If the above catalog table is saved in an ECSV file with the proper units information:
        >>> catalog.write('catalog.ecsv', format='ascii.ecsv')
        >>> cutouts = cutout_tool('h_udf_wfc_b_drz_img.fits', 'catalog.ecsv')

    Parameters
    ----------
    image : str or array or `HDUList` or `PrimaryHDU` or `ImageHDU` or `CompImageHDU`
        Image to cut from. If string is provided, it is assumed to be a
        fits file path.
    catalog : str or `~astropy.table.table.Table`
        Catalog table defining the sources to cut out. Must contain
        unit information as the cutouttool does not assume default units.
        Must be an astropy Table or a file name to an ECSV file containing sources.
    wcs : `~astropy.wcs.wcs.WCS`
        WCS if the input image is an array.
    image_ext : int, optional
        If image is in an HDUList or read from file, use this image extension index
        to extract header and data from the primary image. Default is 0.
    origin : int
        Whether pixel coordinates are 0 or 1-basedpixel coordinates.
    to_fits : bool
        Return cutouts as a list of fits `PrimaryHDU`.
    output_dir : str
        Path to directory to save the cutouts in. If provided, each cutout will be
        saved to a separate file. The directory is created if it does not exist.
    overwrite: bool, optional
        Overwrite existing files. Default is `False`.
    delimiter: str
        Input catalog column delimiter string if reading from a file.
    suppress_rotation : bool
        Suppress rotation even if ``'cutout_pa'`` is provided. Default is `False`.
    verbose : bool, optional
        Print extra info. Default is `True`.

    Returns
    -------
    cutouts : list
        A list of NDData or fits PrimaryHDU. If cutout failed for a target,
       `None` will be added as a place holder.
    """
    # Optional dependencies...
    from reproject.interpolation.high_level import reproject_interp

    save_to_file = output_dir is not None

    # read in the catalog file:
    if isinstance(catalog, str):
        if delimiter is None:
            catalog = QTable.read(catalog)
        else:
            catalog = QTable.read(catalog, delimiter=delimiter)
    elif not isinstance(catalog, Table):
        raise TypeError("Catalog should be an astropy.table.table.Table or"
                        " file name, got {0} instead".format(type(catalog)))

    # Load data and wcs:
    if isinstance(image, np.ndarray):
        # If image is an array type
        if wcs is None:
            raise ValueError("WCS was not provided.")
        data = image
    elif isinstance(image, str):
        # Read data and WCS from file
        with fits.open(image) as pf:
            image_hdu = pf[image_ext]
            data = image_hdu.data
            wcs = WCS(image_hdu.header)
    else:
        # If image is HDUList or HDU:
        if isinstance(image, fits.hdu.hdulist.HDUList):
            image_hdu = image[image_ext]
        elif isinstance(image, (PrimaryHDU, ImageHDU, CompImageHDU)):
            image_hdu = image
        else:
            raise TypeError("Expected array, ImageHDU, HDUList, or file name. Got {0} instead".format(type(image)))
        data = image_hdu.data
        wcs = WCS(image_hdu.header)

    # Calculate the pixel scale of input image:
    pixel_scales = proj_plane_pixel_scales(wcs)
    pixel_scale_width = pixel_scales[0] * u.Unit(wcs.wcs.cunit[0]) / u.pix
    pixel_scale_height = pixel_scales[1] * u.Unit(wcs.wcs.cunit[1]) / u.pix

    # Check if `SkyCoord`s are available:
    if 'ra' in catalog.colnames and 'dec' in catalog.colnames:
        if 'x' in catalog.colnames and 'y' in catalog.colnames:
            raise Exception("Ambiguous catalog: Both (ra, dec) and pixel positions provided.")
        if catalog['ra'].unit is None or catalog['dec'].unit is None:
            raise u.UnitsError("Units not specified for ra and/or dec columns.")
        coords = SkyCoord(catalog['ra'], catalog['dec'], unit=(catalog['ra'].unit,
                                                               catalog['dec'].unit))
    elif 'x' in catalog.colnames and 'y' in catalog.colnames:
        coords = SkyCoord.from_pixel(catalog['x'].astype(float), catalog['y'].astype(float), wcs, origin=origin)
    else:
        try:
            coords = SkyCoord.guess_from_table(catalog)
        except Exception as e:
            raise e

    # Figure out cutout size:
    if 'cutout_width' in catalog.colnames:
        if catalog['cutout_width'].unit is None:
            raise u.UnitsError("Units not specified for cutout_width.")
        if catalog['cutout_width'].unit == u.pix:
            width = catalog['cutout_width'].astype(float)  # pix
        else:
            width = (catalog['cutout_width'] / pixel_scale_width).decompose().value  # pix
    else:
        raise Exception("cutout_width column not found in catalog.")

    if 'cutout_height' in catalog.colnames:
        if catalog['cutout_height'].unit is None:
            raise u.UnitsError("Units not specified for cutout_height.")
        if catalog['cutout_height'].unit == u.pix:
            height = catalog['cutout_height'].astype(float)  # pix
        else:
            height = (catalog['cutout_height'] / pixel_scale_height).decompose().value  # pix
    else:
        raise Exception("cutout_height column not found in catalog.")

    # Do not rotate if column is missing.
    if 'cutout_pa' in catalog.colnames and not suppress_rotation:
        if catalog['cutout_pa'].unit is None:
            raise u.UnitsError("Units not specified for cutout_pa.")
        apply_rotation = True
    else:
        apply_rotation = False

    # Sub-directory, relative to working directory.
    if save_to_file:
        path = output_dir
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

    cutcls = partial(Cutout2D, data, wcs=wcs, mode='partial')
    cutouts = []
    for position, x_pix, y_pix, row in zip(coords, width, height, catalog):
        if apply_rotation:
            pix_rot = row['cutout_pa'].to(u.degree).value

            # Construct new rotated WCS:
            cutout_wcs = WCS(naxis=2)
            cutout_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            cutout_wcs.wcs.crval = [position.ra.deg, position.dec.deg]
            cutout_wcs.wcs.crpix = [(x_pix - 1) * 0.5, (y_pix - 1) * 0.5]

            try:
                cutout_wcs.wcs.cd = wcs.wcs.cd
                cutout_wcs.rotateCD(-pix_rot)
            except AttributeError:
                cutout_wcs.wcs.cdelt = wcs.wcs.cdelt
                cutout_wcs.wcs.crota = [0, -pix_rot]

            cutout_hdr = cutout_wcs.to_header()

            # Rotate the image using reproject
            try:
                cutout_arr = reproject_interp(
                    (data, wcs), cutout_hdr, shape_out=(math.floor(y_pix + math.copysign(0.5, y_pix)),
                        math.floor(x_pix + math.copysign(0.5, x_pix))), order=2)
            except Exception:
                if verbose:
                    log.info('reproject failed: '
                             'Skipping {0}'.format(row['id']))
                cutouts.append(None)
                continue

            cutout_arr = cutout_arr[0]  # Ignore footprint
            cutout_hdr['OBJ_ROT'] = (pix_rot, 'Cutout rotation in degrees')
        else:
            # Make cutout or handle exceptions by adding None to output list
            try:
                cutout = cutcls(position, size=(y_pix, x_pix))
            except NoConvergence:
                if verbose:
                    log.info('WCS solution did not converge: '
                             'Skipping {0}'.format(row['id']))
                cutouts.append(None)
                continue
            except NoOverlapError:
                if verbose:
                    log.info('Cutout is not on image: '
                             'Skipping {0}'.format(row['id']))
                cutouts.append(None)
                continue
            else:
                cutout_hdr = cutout.wcs.to_header()
                cutout_arr = cutout.data

        # If cutout result is empty, skip that target
        if np.array_equiv(cutout_arr, 0):
            if verbose:
                log.info('No data in cutout: Skipping {0}'.format(row['id']))
            cutouts.append(None)
            continue

        # Construct FITS HDU.
        hdu = fits.PrimaryHDU(cutout_arr)
        hdu.header.update(cutout_hdr)
        hdu.header['OBJ_RA'] = (position.ra.deg, 'Cutout object RA in deg')
        hdu.header['OBJ_DEC'] = (position.dec.deg, 'Cutout object DEC in deg')

        # Save to file if output directory is provided
        if save_to_file:
            fname = os.path.join(
                path, '{0}.fits'.format(row['id']))
            try:
                hdu.writeto(fname, overwrite=overwrite)
            except OSError as e:
                if not overwrite:
                    raise OSError(str(e)+" Try setting overwrite parameter to True.")
                else:
                    raise e
            if verbose:
                log.info('Wrote {0}'.format(fname))

        # Add cutout to output list. (as NDData by default)
        if to_fits:
            cutouts.append(hdu)
        else:
            cutouts.append(NDData(data=cutout.data, wcs=cutout.wcs, meta=hdu.header))

    return cutouts


def make_cutouts(catalogname, imagename, image_label, apply_rotation=False,
                 table_format='ascii.ecsv', image_ext=0, clobber=False,
                 verbose=True):
    """Make cutouts from a 2D image and write them to FITS files.

    Catalog must have the following columns with unit info, where applicable:

        * ``'id'`` - ID string; no unit necessary.
        * ``'ra'`` - RA (e.g., in degrees).
        * ``'dec'`` - DEC (e.g., in degrees).
        * ``'cutout_x_size'`` - Cutout width (e.g., in arcsec).
        * ``'cutout_y_size'`` - Cutout height (e.g., in arcsec).
        * ``'cutout_pa'`` - Cutout angle (e.g., in degrees). This is only
          use if user chooses to rotate the cutouts. Positive value
          will result in a clockwise rotation.
        * ``'spatial_pixel_scale'`` - Pixel scale (e.g., in arcsec/pix).

    The following are no longer used, so they are now optional:

        * ``'slit_pa'`` - Slit angle (e.g., in degrees).
        * ``'slit_width'`` - Slit width (e.g., in arcsec).
        * ``'slit_length'`` - Slit length (e.g., in arcsec).

    Cutouts are organized as follows::

        working_dir/
            <image_label>_cutouts/
                <id>_<image_label>_cutout.fits

    Each cutout image is a simple single-extension FITS with updated WCS.
    Its header has the following special keywords:

        * ``OBJ_RA`` - RA of the cutout object in degrees.
        * ``OBJ_DEC`` - DEC of the cutout object in degrees.
        * ``OBJ_ROT`` - Rotation of cutout object in degrees.

    Parameters
    ----------
    catalogname : str
        Catalog table defining the sources to cut out.

    imagename : str
        Image to cut.

    image_label : str
        Label to name the cutout sub-directory and filenames.

    apply_rotation : bool
        Cutout will be rotated to a given angle. Default is `False`.

    table_format : str, optional
        Format as accepted by `~astropy.table.QTable`. Default is ECSV.

    image_ext : int, optional
        Image extension to extract header and data. Default is 0.

    clobber : bool, optional
        Overwrite existing files. Default is `False`.

    verbose : bool, optional
        Print extra info. Default is `True`.

    """
    # Optional dependencies...
    from reproject import reproject_interp

    table = QTable.read(catalogname, format=table_format)

    with fits.open(imagename) as pf:
        data = pf[image_ext].data
        wcs = WCS(pf[image_ext].header)

    # It is more efficient to operate on an entire column at once.
    c = SkyCoord(table['ra'], table['dec'])
    x = (table['cutout_x_size'] / table['spatial_pixel_scale']).value  # pix
    y = (table['cutout_y_size'] / table['spatial_pixel_scale']).value  # pix
    pscl = table['spatial_pixel_scale'].to(u.deg / u.pix)

    # Do not rotate if column is missing.
    if 'cutout_pa' not in table.colnames:
        apply_rotation = False

    # Sub-directory, relative to working directory.
    path = '{0}_cutouts'.format(image_label)
    if not os.path.exists(path):
        os.mkdir(path)

    cutcls = partial(Cutout2D, data, wcs=wcs, mode='partial')

    for position, x_pix, y_pix, pix_scl, row in zip(c, x, y, pscl, table):

        if apply_rotation:
            pix_rot = row['cutout_pa'].to(u.degree).value

            cutout_wcs = WCS(naxis=2)
            cutout_wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            cutout_wcs.wcs.crval = [position.ra.deg, position.dec.deg]
            cutout_wcs.wcs.crpix = [(x_pix - 1) * 0.5, (y_pix - 1) * 0.5]

            try:
                cutout_wcs.wcs.cd = wcs.wcs.cd
                cutout_wcs.rotateCD(-pix_rot)
            except AttributeError:
                cutout_wcs.wcs.cdelt = wcs.wcs.cdelt
                cutout_wcs.wcs.crota = [0, -pix_rot]

            cutout_hdr = cutout_wcs.to_header()

            try:
                cutout_arr = reproject_interp(
                    (data, wcs), cutout_hdr, shape_out=(math.floor(y_pix + math.copysign(0.5, y_pix)),
                        math.floor(x_pix + math.copysign(0.5, x_pix))), order=2)
            except Exception:
                if verbose:
                    log.info('reproject failed: '
                             'Skipping {0}'.format(row['id']))
                continue

            cutout_arr = cutout_arr[0]  # Ignore footprint
            cutout_hdr['OBJ_ROT'] = (pix_rot, 'Cutout rotation in degrees')

        else:
            try:
                cutout = cutcls(position, size=(y_pix, x_pix))
            except NoConvergence:
                if verbose:
                    log.info('WCS solution did not converge: '
                             'Skipping {0}'.format(row['id']))
                continue
            except NoOverlapError:
                if verbose:
                    log.info('Cutout is not on image: '
                             'Skipping {0}'.format(row['id']))
                continue
            else:
                cutout_hdr = cutout.wcs.to_header()
                cutout_arr = cutout.data

        if np.array_equiv(cutout_arr, 0):
            if verbose:
                log.info('No data in cutout: Skipping {0}'.format(row['id']))
            continue

        fname = os.path.join(
            path, '{0}_{1}_cutout.fits'.format(row['id'], image_label))

        # Construct FITS HDU.
        hdu = fits.PrimaryHDU(cutout_arr)
        hdu.header.update(cutout_hdr)
        hdu.header['OBJ_RA'] = (position.ra.deg, 'Cutout object RA in deg')
        hdu.header['OBJ_DEC'] = (position.dec.deg, 'Cutout object DEC in deg')

        hdu.writeto(fname, clobber=clobber)

        if verbose:
            log.info('Wrote {0}'.format(fname))


def show_cutout_with_slit(hdr, data=None, slit_ra=None, slit_dec=None,
                          slit_shape='rectangular', slit_width=0.2,
                          slit_length=3.3, slit_angle=90, slit_radius=0.2,
                          slit_rout=0.5, cmap='Greys_r', plotname='',
                          **kwargs):
    """Show a cutout image with the slit(s) superimposed.

    Parameters
    ----------
    hdr : dict
        Cutout image header.

    data : ndarray or `None`, optional
        Cutout image data. If not given, data is not shown.

    slit_ra, slit_dec : float or array or `None`, optional
        Slit RA and DEC in degrees. Default is to use object position
        from image header. If an array is given, each pair of RA and
        DEC becomes a slit.

    slit_shape : {'annulus', 'circular', 'rectangular'}, optional
        Shape of the slit (circular or rectangular).
        Default is rectangular.

    slit_width, slit_length : float, optional
        Rectangular slit width and length in arcseconds.
        Defaults are some fudge values.

    slit_angle : float, optional
        Rectangular slit angle in degrees for the display.
        Default is vertical.

    slit_radius : float, optional
        Radius of a circular or annulus slit in arcseconds.
        For annulus, this is the inner radius.
        Default is some fudge value.

    slit_rout : float, optional
        Outer radius of an annulus slit in arcseconds.
        Default is some fudge value.

    cmap : str or obj, optional
        Matplotlib color map for image display. Default is grayscale.

    plotname : str, optional
        Filename to save plot as. If not given, it is not saved.

    kwargs : dict, optional
        Keyword argument(s) for the aperture overlay.
        If ``ax`` is given, it will also be used for image display.

    See Also
    --------
    make_cutouts

    """
    # Optional dependencies...
    import matplotlib.pyplot as plt
    from photutils import (SkyCircularAnnulus, SkyCircularAperture,
                           SkyRectangularAperture)

    if slit_ra is None:
        slit_ra = hdr['OBJ_RA']
    if slit_dec is None:
        slit_dec = hdr['OBJ_DEC']

    position = SkyCoord(slit_ra, slit_dec, unit='deg')

    if slit_shape == 'circular':
        slit_radius = u.Quantity(slit_radius, u.arcsec)
        aper = SkyCircularAperture(position, slit_radius)

    elif slit_shape == 'annulus':
        slit_rin = u.Quantity(slit_radius, u.arcsec)
        slit_rout = u.Quantity(slit_rout, u.arcsec)
        aper = SkyCircularAnnulus(position, slit_rin, slit_rout)

    else:  # rectangular
        slit_width = u.Quantity(slit_width, u.arcsec)
        slit_length = u.Quantity(slit_length, u.arcsec)
        slit_angle = u.Quantity(slit_angle, u.degree)
        aper = SkyRectangularAperture(position, slit_width, slit_length,
                                      theta=slit_angle)

    wcs = WCS(hdr)
    aper_pix = aper.to_pixel(wcs)
    ax = kwargs.get('ax', plt)

    if data is not None:
        ax.imshow(data, cmap=cmap, origin='lower')

    aper_pix.plot(**kwargs)

    if plotname:
        ax.savefig(plotname)
