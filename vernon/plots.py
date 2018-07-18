# -*- mode: python; coding: utf-8 -*-
# Copyright 2016 Peter Williams and collaborators.
# Licensed under the MIT License.

"""
Helpers for plotting.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np, omega as om


__all__ = str('''
pangosub
figdefaults
BigArrow
KeyHighlightStamp
ManualStampKeyPainter
Rectangle
XYText
''').split ()


def pangosub (s):
    return (s
            .replace ('{^', '<span size="smaller" rise="2000">')
            .replace ('{', '<span size="smaller" rise="-2000">')
            .replace ('}', '</span>'))


figdefaults = {
    'omstyle': om.styles.ColorOnWhiteVector (),
    #'pangofamily': 'Linux Biolinum O',
}


class KeyHighlightStamp (om.stamps.Circle):
    def __init__ (self, box=False):
        super (KeyHighlightStamp, self).__init__ (size=3, fill=True)
        self._box = box

    def _paintOne (self, ctxt, style, size):
        if self._box:
            om.stamps.symBox (ctxt, style, size, self.fill)
        else:
            om.stamps.symCircle (ctxt, style, size, self.fill)

        ctxt.set_source_rgb (0, 0, 0)
        om.stamps.symCircle (ctxt, style, 5, False)


class ManualStampKeyPainter (om.rect.GenericKeyPainter):
    def __init__ (self, keytext, stamp, stampstyle=None):
        self.keytext = keytext
        self.stamp = stamp
        self.stampStyle = stampstyle
        stamp.setData ('no data allowed for key-only stamp')

    def _getText (self):
        return self.keytext

    def _drawLine (self):
        return False

    def _drawStamp (self):
        return True

    def _drawRegion (self):
        return False

    def _applyStampStyle (self, style, ctxt):
        style.apply (ctxt, self.stampStyle)

    def _getStamp (self):
        return self.stamp


class BigArrow (om.rect.FieldPainter):
    needsDataStyle = False
    dsn = None
    style = {'color': 'foreground', 'linewidth': 2}
    length = 36
    headsize = 16
    direction = None

    def __init__ (self, x, y, direction):
        super (BigArrow, self).__init__ ()
        self.x = float (x)
        self.y = float (y)
        self.direction = direction

    def getDataBounds (self):
        return self.x, self.x, self.y, self.y

    def getKeyPainter (self):
        return None

    def doPaint (self, ctxt, style):
        super (BigArrow, self).doPaint (ctxt, style)

        x = self.xform.mapX (self.x)
        y = self.xform.mapY (self.y)

        ctxt.save ()
        style.apply (ctxt, self.style)
        om.stamps.arrow (ctxt, x, y, self.direction, self.length, self.headsize)
        ctxt.restore ()


class XYText (om.rect.FieldPainter):
    """Paint text at an X/Y location on a plot.

    The precise position is set by the combination of the `x`, `y`, `hAnchor`,
    and `vAnchor` parameters. `x` and `y` specify a location on the plot.
    `hAnchor` and `vAnchor` specify where the text lands relative to the
    location. `hAnchor = 0` means that the left edge of the text is found at
    `x`; `vAnchor = 1` means that the bottom edge of the text is found at `y`;
    and so on. The default setting of `hAnchor = vAnchor = 0.5` means that the
    exact center of the text is positioned at `x, y`.

    """
    style = None
    color = (0, 0, 0)
    hAnchor = 0.5
    vAnchor = 0.5
    needsDataStyle = False
    dsn = None
    _ts = None

    def __init__ (self, x, y, text, hAnchor=0.5, vAnchor=0.5):
        super (XYText, self).__init__ ()
        self.text = unicode (text)
        self.x = float (x)
        self.y = float (y)
        self.hAnchor = float (hAnchor)
        self.vAnchor = float (vAnchor)

    def getDataBounds (self):
        return self.x, self.x, self.y, self.y

    def getKeyPainter (self):
        return None

    def doPaint (self, ctxt, style):
        super (XYText, self).doPaint (ctxt, style)

        if self._ts is None:
            self._ts = om.TextStamper (self.text)
        x = self.xform.mapX (self.x)
        y = self.xform.mapY (self.y)
        w, h = self._ts.getSize (ctxt, style)

        ctxt.save ()
        style.apply (ctxt, self.style)
        self._ts.paintAt (ctxt,
                          x - self.hAnchor * w,
                          y - self.vAnchor * h,
                          self.color)
        ctxt.restore ()


class Rectangle (om.rect.FieldPainter):
    """Plot a rectangle."""

    stroke = True
    fill = False
    style = 'bgLinework'

    def __init__ (self, x0, x1, y0, y1, stroke=True, fill=False, style='bgLinework'):
        super (Rectangle, self).__init__ ()

        self.stroke = stroke
        self.fill = fill
        self.style = style

        if x0 > x1:
            x0, x1 = x1, x0
        self.xmin, self.xmax = x0, x1

        if y0 > y1:
            y0, y1 = y1, y0
        self.ymin, self.ymax = y0, y1

    def getDataBounds (self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    def getKeyPainter (self):
        return None

    def doPaint (self, ctxt, style):
        super (Rectangle, self).doPaint (ctxt, style)

        x0, x1 = self.xform.mapX (np.asarray ([self.xmin, self.xmax]))
        y0, y1 = self.xform.mapY (np.asarray ([self.ymin, self.ymax]))

        # We may not have x1 > x0, depending on the axis transform.
        x = min (x0, x1)
        y = min (y0, y1)
        w = abs (x1 - x0)
        h = abs (y1 - y0)

        ctxt.save ()
        style.apply (ctxt, self.style)
        ctxt.rectangle (x, y, w, h)
        if self.fill:
            ctxt.fill_preserve ()
        if self.stroke:
            ctxt.stroke ()
        ctxt.new_path () # clear path if we didn't stroke; restore() doesn't!
        ctxt.restore ()
