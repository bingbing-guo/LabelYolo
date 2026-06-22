#include "canvaswidget.h"

#include <QLineF>
#include <QMouseEvent>
#include <QPainter>
#include <QUndoCommand>
#include <QtMath>

namespace {
class AnnotationSnapshotCommand final : public QUndoCommand
{
public:
    AnnotationSnapshotCommand(CanvasWidget *canvas,
                              const QList<Annotation> &before, int beforeSelection,
                              const QList<Annotation> &after, int afterSelection,
                              const QString &text)
        : QUndoCommand(text), m_canvas(canvas), m_before(before), m_after(after),
          m_beforeSelection(beforeSelection), m_afterSelection(afterSelection) {}
    void undo() override { m_canvas->applyAnnotationSnapshot(m_before, m_beforeSelection); }
    void redo() override { m_canvas->applyAnnotationSnapshot(m_after, m_afterSelection); }
private:
    CanvasWidget *m_canvas;
    QList<Annotation> m_before;
    QList<Annotation> m_after;
    int m_beforeSelection;
    int m_afterSelection;
};
}

CanvasWidget::CanvasWidget(QWidget *parent) : QWidget(parent)
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
}

void CanvasWidget::setImage(const QImage &img)
{
    m_originalImage = img;
    resetView();
    m_undoStack.clear();
    update();
}

void CanvasWidget::resetView()
{
    m_scale = 1.0;
    m_rotateAngle = 0;
    m_translate = QPointF(width() / 2.0, height() / 2.0);
    emit viewChanged(m_scale);
}

void CanvasWidget::rotateClockwise()
{
    m_rotateAngle = (m_rotateAngle + 90) % 360;
    update();
}

QPointF CanvasWidget::mapToImage(const QPointF &mousePos) const
{
    QTransform transform;
    transform.translate(m_translate.x(), m_translate.y());
    transform.rotate(m_rotateAngle);
    transform.scale(m_scale, m_scale);
    return transform.inverted().map(mousePos);
}

int CanvasWidget::getAnnotationAt(const QPointF &pos) const
{
    const QPointF imagePos = mapToImage(pos);
    for (int i = m_annotations.size() - 1; i >= 0; --i)
        if (m_annotations[i].rect.contains(imagePos)) return i;
    return -1;
}

CanvasWidget::ResizeHandle CanvasWidget::getResizeHandleAt(const QPointF &pos) const
{
    if (m_selectedIndex < 0 || m_selectedIndex >= m_annotations.size()) return NoHandle;
    const QRectF r = m_annotations[m_selectedIndex].rect;
    const QPointF p = mapToImage(pos);
    const qreal distance = 8.0 / qMax<qreal>(m_scale, 0.1);
    const QList<QPair<QPointF, ResizeHandle>> handles = {
        {r.topLeft(), TopLeft}, {QPointF(r.center().x(), r.top()), Top}, {r.topRight(), TopRight},
        {QPointF(r.right(), r.center().y()), Right}, {r.bottomRight(), BottomRight},
        {QPointF(r.center().x(), r.bottom()), Bottom}, {r.bottomLeft(), BottomLeft},
        {QPointF(r.left(), r.center().y()), Left}
    };
    for (const auto &handle : handles)
        if (QLineF(p, handle.first).length() <= distance) return handle.second;
    return NoHandle;
}

QRectF CanvasWidget::boundedRect(const QRectF &rect) const
{
    const QRectF bounds(-m_originalImage.width() / 2.0, -m_originalImage.height() / 2.0,
                        m_originalImage.width(), m_originalImage.height());
    QRectF result = rect.normalized().intersected(bounds);
    constexpr qreal minimum = 4.0;
    if (result.width() < minimum) result.setWidth(minimum);
    if (result.height() < minimum) result.setHeight(minimum);
    if (result.right() > bounds.right()) result.moveRight(bounds.right());
    if (result.bottom() > bounds.bottom()) result.moveBottom(bounds.bottom());
    return result;
}

void CanvasWidget::setSelectedIndex(int index)
{
    if (index < -1 || index >= m_annotations.size()) index = -1;
    for (int i = 0; i < m_annotations.size(); ++i) m_annotations[i].isSelected = (i == index);
    m_selectedIndex = index;
    emit selectionChanged(index);
    update();
}

void CanvasWidget::selectAnnotation(int index) { setSelectedIndex(index); }

void CanvasWidget::applyAnnotationSnapshot(const QList<Annotation> &annotations, int selectedIndex)
{
    m_replayingSnapshot = true;
    m_annotations = annotations;
    setSelectedIndex(selectedIndex);
    emit annotationsChanged();
    m_replayingSnapshot = false;
}

void CanvasWidget::pushSnapshot(const QList<Annotation> &before, int beforeSelection, const QString &text)
{
    m_undoStack.push(new AnnotationSnapshotCommand(this, before, beforeSelection,
                                                    m_annotations, m_selectedIndex, text));
}

void CanvasWidget::changeSelectedClass(const QString &name, int id)
{
    if (m_replayingSnapshot) return;
    if (m_selectedIndex < 0 || m_selectedIndex >= m_annotations.size()) return;
    if (m_annotations[m_selectedIndex].classId == id && m_annotations[m_selectedIndex].className == name) return;
    const auto before = m_annotations;
    const int beforeSelection = m_selectedIndex;
    m_annotations[m_selectedIndex].classId = id;
    m_annotations[m_selectedIndex].className = name;
    pushSnapshot(before, beforeSelection, tr("Change annotation class"));
}

void CanvasWidget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::lightGray);
    if (m_originalImage.isNull()) return;
    painter.setRenderHint(QPainter::Antialiasing);
    painter.save();
    painter.translate(m_translate);
    painter.rotate(m_rotateAngle);
    painter.scale(m_scale, m_scale);
    painter.drawImage(QPointF(-m_originalImage.width() / 2.0, -m_originalImage.height() / 2.0), m_originalImage);

    for (const auto &annotation : m_annotations) {
        QPen pen(annotation.isSelected ? Qt::red : Qt::green);
        pen.setWidthF(2.0 / m_scale);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(annotation.rect);
        painter.drawText(annotation.rect.topLeft() + QPointF(0, -2), annotation.className);
        if (annotation.isSelected) {
            const qreal size = 8.0 / m_scale;
            const QList<QPointF> points = {annotation.rect.topLeft(), QPointF(annotation.rect.center().x(), annotation.rect.top()),
                annotation.rect.topRight(), QPointF(annotation.rect.right(), annotation.rect.center().y()), annotation.rect.bottomRight(),
                QPointF(annotation.rect.center().x(), annotation.rect.bottom()), annotation.rect.bottomLeft(),
                QPointF(annotation.rect.left(), annotation.rect.center().y())};
            painter.setBrush(Qt::white);
            for (const QPointF &point : points)
                painter.drawRect(QRectF(point.x() - size / 2, point.y() - size / 2, size, size));
        }
    }
    if (m_isCreating && !m_tempRect.isNull()) {
        QPen pen(Qt::yellow); pen.setWidthF(2.0 / m_scale); painter.setPen(pen);
        painter.setBrush(Qt::NoBrush); painter.drawRect(m_tempRect.normalized());
    }
    painter.restore();
}

void CanvasWidget::mousePressEvent(QMouseEvent *event)
{
    if (m_originalImage.isNull()) return;
    if (event->button() == Qt::LeftButton) {
        m_gestureBefore = m_annotations;
        m_gestureSelectionBefore = m_selectedIndex;
        m_resizeHandle = getResizeHandleAt(event->position());
        if (m_resizeHandle != NoHandle) {
            m_isResizingBox = true;
        } else {
            const int hit = getAnnotationAt(event->position());
            setSelectedIndex(hit);
            if (hit >= 0) m_isMovingBox = true;
            else {
                m_isCreating = true;
                const QPointF imagePos = mapToImage(event->position());
                m_tempRect = QRectF(imagePos, imagePos);
            }
        }
        m_lastMousePos = event->position();
    } else if (event->button() == Qt::RightButton) {
        m_isPanning = true;
        m_lastMousePos = event->position();
        setCursor(Qt::ClosedHandCursor);
    }
    update();
}

void CanvasWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (m_originalImage.isNull()) return;
    if (m_isCreating) {
        m_tempRect.setBottomRight(mapToImage(event->position()));
    } else if (m_isResizingBox && m_selectedIndex >= 0) {
        const QPointF p = mapToImage(event->position());
        QRectF r = m_annotations[m_selectedIndex].rect;
        if (m_resizeHandle == TopLeft || m_resizeHandle == Left || m_resizeHandle == BottomLeft) r.setLeft(p.x());
        if (m_resizeHandle == TopRight || m_resizeHandle == Right || m_resizeHandle == BottomRight) r.setRight(p.x());
        if (m_resizeHandle == TopLeft || m_resizeHandle == Top || m_resizeHandle == TopRight) r.setTop(p.y());
        if (m_resizeHandle == BottomLeft || m_resizeHandle == Bottom || m_resizeHandle == BottomRight) r.setBottom(p.y());
        m_annotations[m_selectedIndex].rect = boundedRect(r);
    } else if (m_isMovingBox && m_selectedIndex >= 0) {
        const QPointF delta = mapToImage(event->position()) - mapToImage(m_lastMousePos);
        m_annotations[m_selectedIndex].rect = boundedRect(m_annotations[m_selectedIndex].rect.translated(delta));
    } else if (m_isPanning) {
        m_translate += event->position() - m_lastMousePos;
    }
    m_lastMousePos = event->position();
    update();
}

void CanvasWidget::mouseReleaseEvent(QMouseEvent *)
{
    QString action;
    if (m_isCreating) {
        const QRectF draggedRect = m_tempRect.normalized();
        if (draggedRect.width() >= 4 && draggedRect.height() >= 4) {
            const QRectF rect = boundedRect(draggedRect);
            Annotation annotation;
            annotation.rect = rect;
            annotation.classId = m_currentClassId;
            annotation.className = m_currentClassName;
            m_annotations.append(annotation);
            setSelectedIndex(m_annotations.size() - 1);
            action = tr("Create annotation");
        }
    } else if (m_isMovingBox) action = tr("Move annotation");
    else if (m_isResizingBox) action = tr("Resize annotation");

    m_isCreating = m_isMovingBox = m_isPanning = m_isResizingBox = false;
    m_resizeHandle = NoHandle;
    m_tempRect = QRectF();
    setCursor(Qt::ArrowCursor);
    if (!action.isEmpty() && m_gestureBefore != m_annotations)
        pushSnapshot(m_gestureBefore, m_gestureSelectionBefore, action);
    update();
}

void CanvasWidget::wheelEvent(QWheelEvent *event)
{
    if (m_originalImage.isNull()) return;
    const QPointF mouse = event->position();
    const QPointF imageBefore = mapToImage(mouse);
    m_scale = qBound<qreal>(0.05, m_scale * (event->angleDelta().y() > 0 ? 1.1 : 0.9), 30.0);
    QTransform rotation;
    rotation.rotate(m_rotateAngle);
    m_translate = mouse - rotation.map(imageBefore * m_scale);
    emit viewChanged(m_scale);
    update();
}

void CanvasWidget::deleteSelectedAnnotation()
{
    if (m_selectedIndex < 0 || m_selectedIndex >= m_annotations.size()) return;
    const auto before = m_annotations;
    const int beforeSelection = m_selectedIndex;
    m_annotations.removeAt(m_selectedIndex);
    setSelectedIndex(-1);
    pushSnapshot(before, beforeSelection, tr("Delete annotation"));
}

void CanvasWidget::clearAnnotations()
{
    if (m_annotations.isEmpty()) return;
    const auto before = m_annotations;
    const int beforeSelection = m_selectedIndex;
    m_annotations.clear();
    m_selectedIndex = -1;
    pushSnapshot(before, beforeSelection, tr("Redo annotation"));
}

void CanvasWidget::loadAnnotations(const QList<Annotation> &annotations)
{
    m_annotations = annotations;
    m_undoStack.clear();
    setSelectedIndex(-1);
    emit annotationsChanged();
}

void CanvasWidget::addAnnotationClass(const QString &name, int id)
{
    m_currentClassName = name;
    m_currentClassId = id;
}
