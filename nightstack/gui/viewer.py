"""Zoomable photo viewer with the mask overlay, red star dots, green horizon points and brushes."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsView)


class Tool(Enum):
    PAN = 0
    ERASE = 1
    SKY = 2
    ADD_STAR = 3
    HORIZON = 4


def to_qimage(rgb: np.ndarray) -> QImage:
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, 2)
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def overlay_qimage(mask: np.ndarray, color=(255, 120, 40), alpha: int = 95) -> QImage:
    """Tint the foreground (mask <= 127) so the sky stays clear to judge."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    g = mask <= 127
    rgba[g] = (*color, alpha)
    # Thin bright line at the boundary helps judge the edge.
    edge = g ^ np.roll(g, 1, 0) | g ^ np.roll(g, 1, 1)
    rgba[edge] = (255, 200, 80, 230)
    return QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


class PointsItem(QGraphicsItem):
    """Many dots drawn at a constant on-screen size (fast even with thousands)."""

    def __init__(self, color: QColor, size: float):
        super().__init__()
        self.poly = QPolygonF()
        self.rect = QRectF()
        self.pen = QPen(color, size, Qt.SolidLine, Qt.RoundCap)
        self.pen.setCosmetic(True)
        self.setZValue(3)

    def set_points(self, pts: np.ndarray, rect: QRectF):
        self.prepareGeometryChange()
        self.poly = QPolygonF([QPointF(float(x), float(y)) for x, y in pts])
        self.rect = rect
        self.update()

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(self.pen)
        painter.drawPoints(self.poly)


class HorizonItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.points: List[tuple] = []
        self.rect = QRectF()
        self.setZValue(4)
        self.line_pen = QPen(QColor(80, 255, 140, 220), 2)
        self.line_pen.setCosmetic(True)
        self.dot_pen = QPen(QColor(80, 255, 140), 10, Qt.SolidLine, Qt.RoundCap)
        self.dot_pen.setCosmetic(True)
        self.inner_pen = QPen(QColor(10, 40, 20), 4, Qt.SolidLine, Qt.RoundCap)
        self.inner_pen.setCosmetic(True)

    def set_points(self, pts, rect):
        self.prepareGeometryChange()
        self.points = sorted(pts)
        self.rect = rect
        self.update()

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget=None):
        if not self.points:
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        pts = self.points
        path = QPainterPath(QPointF(0, pts[0][1]))
        for x, y in pts:
            path.lineTo(x, y)
        path.lineTo(self.rect.width(), pts[-1][1])
        painter.setPen(self.line_pen)
        painter.drawPath(path)
        poly = QPolygonF([QPointF(x, y) for x, y in pts])
        painter.setPen(self.dot_pen)
        painter.drawPoints(poly)
        painter.setPen(self.inner_pen)
        painter.drawPoints(poly)


class ImageViewer(QGraphicsView):
    stroke = Signal(float, float, object, bool)   # x, y, last point or None, is_first
    strokeEnded = Signal()
    starClicked = Signal(float, float)
    horizonEdited = Signal(list)
    horizonEditStarted = Signal()
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setAcceptDrops(False)
        self.pix = QGraphicsPixmapItem()
        self.pix.setTransformationMode(Qt.SmoothTransformation)
        self.overlay = QGraphicsPixmapItem()
        self.overlay.setZValue(2)
        self.stars = PointsItem(QColor(255, 60, 60), 5)
        self.horizon = HorizonItem()
        self.cursor_ring = QGraphicsEllipseItem()
        pen = QPen(QColor(255, 255, 255, 200), 1.5)
        pen.setCosmetic(True)
        self.cursor_ring.setPen(pen)
        self.cursor_ring.setZValue(10)
        self.cursor_ring.hide()
        for it in (self.pix, self.overlay, self.stars, self.horizon, self.cursor_ring):
            self.scene().addItem(it)
        self.tool = Tool.PAN
        self.brush = 30.0
        self._last = None
        self._drag_handle: Optional[int] = None
        self._space = False
        self._panning = False
        self._pan_start = None
        self._has_image = False
        self.edit_enabled = False
        self.set_tool(Tool.PAN)

    # ---------------------------------------------------------------- content
    def set_image(self, rgb: np.ndarray, keep_view: bool = False):
        qimg = to_qimage(rgb)
        first = not self._has_image or self.pix.pixmap().size() != qimg.size()
        self.pix.setPixmap(QPixmap.fromImage(qimg))
        self.scene().setSceneRect(QRectF(0, 0, qimg.width(), qimg.height()))
        self._has_image = True
        if first and not keep_view:
            self.fit()

    def image_rect(self) -> QRectF:
        return QRectF(self.pix.pixmap().rect())

    def set_overlay(self, mask: Optional[np.ndarray], opacity: float = 1.0):
        if mask is None:
            self.overlay.hide()
            return
        self.overlay.setPixmap(QPixmap.fromImage(overlay_qimage(mask)))
        self.overlay.setOpacity(opacity)
        self.overlay.show()

    def set_stars(self, pts: Optional[np.ndarray]):
        if pts is None:
            self.stars.hide()
            return
        self.stars.set_points(pts, self.image_rect())
        self.stars.show()

    def set_horizon(self, pts: Optional[list]):
        if pts is None:
            self.horizon.hide()
            return
        self.horizon.set_points(pts, self.image_rect())
        self.horizon.show()

    def clear_overlays(self):
        self.overlay.hide()
        self.stars.hide()
        self.horizon.hide()

    # ---------------------------------------------------------------- tools & zoom
    def set_tool(self, tool: Tool):
        self.tool = tool
        brush = tool in (Tool.ERASE, Tool.SKY) and self.edit_enabled
        self.cursor_ring.setVisible(brush)
        if tool == Tool.PAN or not self.edit_enabled:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.BlankCursor if brush else Qt.CrossCursor)
        self._update_ring()

    def set_brush(self, r: float):
        self.brush = r
        self._update_ring()

    def _update_ring(self, pos: Optional[QPointF] = None):
        if pos is None:
            c = self.cursor_ring.rect().center() if not self.cursor_ring.rect().isNull() else QPointF(0, 0)
        else:
            c = pos
        r = self.brush
        self.cursor_ring.setRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))

    def fit(self):
        if not self._has_image:
            return
        self.fitInView(self.image_rect(), Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.transform().m11())

    def zoom_100(self, full_res_scale: float = 1.0):
        self.resetTransform()
        self.scale(1 / full_res_scale, 1 / full_res_scale)
        self.zoomChanged.emit(self.transform().m11())

    def zoom_by(self, f: float):
        cur = self.transform().m11()
        f = max(0.02 / cur, min(f, 40 / cur))
        self.scale(f, f)
        self.zoomChanged.emit(self.transform().m11())

    def wheelEvent(self, e):
        # Trackpad two-finger scroll pans; mouse wheel or Cmd/Ctrl+scroll zooms.
        if not e.pixelDelta().isNull() and not (e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)):
            return super().wheelEvent(e)
        dy = e.angleDelta().y()
        if dy:
            self.zoom_by(1.0015 ** dy)

    def viewportEvent(self, e):
        if e.type() == e.Type.NativeGesture:
            try:
                if e.gestureType() == Qt.ZoomNativeGesture:
                    self.zoom_by(1 + e.value())
                    return True
            except AttributeError:
                pass
        return super().viewportEvent(e)

    # ---------------------------------------------------------------- mouse
    def _handle_at(self, view_pos) -> Optional[int]:
        best, bd = None, 12.0
        for i, (x, y) in enumerate(self.horizon.points):
            vp = self.mapFromScene(QPointF(x, y))
            d = ((vp.x() - view_pos.x()) ** 2 + (vp.y() - view_pos.y()) ** 2) ** 0.5
            if d < bd:
                best, bd = i, d
        return best

    def _pan_active(self, e) -> bool:
        return self._space or e.buttons() & Qt.MiddleButton or self.tool == Tool.PAN or not self.edit_enabled

    def mousePressEvent(self, e):
        if (e.button() == Qt.MiddleButton) or (self._space and e.button() == Qt.LeftButton):
            self._panning = True
            self._pan_start = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if self.tool == Tool.PAN or not self.edit_enabled:
            return super().mousePressEvent(e)
        sp = self.mapToScene(e.position().toPoint())
        if self.tool in (Tool.ERASE, Tool.SKY) and e.button() == Qt.LeftButton:
            self._last = (sp.x(), sp.y())
            self.stroke.emit(sp.x(), sp.y(), None, True)
        elif self.tool == Tool.ADD_STAR and e.button() == Qt.LeftButton:
            self.starClicked.emit(sp.x(), sp.y())
        elif self.tool == Tool.HORIZON:
            h = self._handle_at(e.position())
            delete = e.button() == Qt.RightButton or (e.modifiers() & Qt.AltModifier)
            if h is not None and delete:
                self.horizonEditStarted.emit()
                pts = list(self.horizon.points)
                pts.pop(h)
                self.horizon.set_points(pts, self.image_rect())
                self.horizonEdited.emit(pts)
            elif h is not None:
                self.horizonEditStarted.emit()
                self._drag_handle = h
            elif e.button() == Qt.LeftButton:
                self.horizonEditStarted.emit()
                r = self.image_rect()
                pts = list(self.horizon.points) + [(min(max(sp.x(), 0), r.width()), min(max(sp.y(), 0), r.height()))]
                self.horizon.set_points(pts, r)
                self._drag_handle = self.horizon.points.index(pts[-1])
                self.horizonEdited.emit(self.horizon.points)

    def mouseMoveEvent(self, e):
        sp = self.mapToScene(e.position().toPoint())
        if self._panning and self._pan_start is not None:
            d = e.position() - self._pan_start
            self._pan_start = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        self._update_ring(sp)
        if self.tool in (Tool.ERASE, Tool.SKY) and self.edit_enabled and e.buttons() & Qt.LeftButton \
                and self._last is not None:
            self.stroke.emit(sp.x(), sp.y(), self._last, False)
            self._last = (sp.x(), sp.y())
            return
        if self.tool == Tool.HORIZON and self._drag_handle is not None and e.buttons() & Qt.LeftButton:
            r = self.image_rect()
            pts = list(self.horizon.points)
            pts[self._drag_handle] = (min(max(sp.x(), 0), r.width()), min(max(sp.y(), 0), r.height()))
            moved = pts[self._drag_handle]
            self.horizon.set_points(pts, r)
            self._drag_handle = self.horizon.points.index(moved)
            self.horizonEdited.emit(self.horizon.points)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._panning:
            self._panning = False
            self.set_tool(self.tool)
            return
        if self._last is not None:
            self._last = None
            self.strokeEnded.emit()
        self._drag_handle = None
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.set_tool(self.tool)
            return
        super().keyReleaseEvent(e)

    def leaveEvent(self, e):
        self.cursor_ring.hide()
        super().leaveEvent(e)

    def enterEvent(self, e):
        self.cursor_ring.setVisible(self.tool in (Tool.ERASE, Tool.SKY) and self.edit_enabled)
        super().enterEvent(e)
