#ifndef CANVASWIDGET_H
#define CANVASWIDGET_H

#include <QImage>
#include <QList>
#include <QRectF>
#include <QTransform>
#include <QUndoStack>
#include <QWidget>

struct Annotation {
    QRectF rect;
    int classId = 0;
    QString className;
    bool isSelected = false;

    bool operator==(const Annotation &other) const {
        return rect == other.rect && classId == other.classId &&
               className == other.className && isSelected == other.isSelected;
    }
};

class CanvasWidget : public QWidget
{
    Q_OBJECT
public:
    explicit CanvasWidget(QWidget *parent = nullptr);

    void setImage(const QImage &img);
    void resetView();
    void rotateClockwise();
    void addAnnotationClass(const QString &name, int id);
    void deleteSelectedAnnotation();
    void clearAnnotations();
    QList<Annotation> getAnnotations() const { return m_annotations; }
    void loadAnnotations(const QList<Annotation> &anns);
    void selectAnnotation(int index);
    void changeSelectedClass(const QString &name, int id);
    int selectedIndex() const { return m_selectedIndex; }
    QUndoStack *undoStack() { return &m_undoStack; }
    QSize getImageSize() const { return m_originalImage.size(); }
    void applyAnnotationSnapshot(const QList<Annotation> &annotations, int selectedIndex);

signals:
    void annotationsChanged();
    void selectionChanged(int index);
    void viewChanged(qreal scale);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    enum ResizeHandle { NoHandle, TopLeft, Top, TopRight, Right, BottomRight, Bottom, BottomLeft, Left };
    QPointF mapToImage(const QPointF &mousePos) const;
    int getAnnotationAt(const QPointF &pos) const;
    ResizeHandle getResizeHandleAt(const QPointF &pos) const;
    QRectF boundedRect(const QRectF &rect) const;
    void setSelectedIndex(int index);
    void pushSnapshot(const QList<Annotation> &before, int beforeSelection, const QString &text);

    QImage m_originalImage;
    qreal m_scale = 1.0;
    int m_rotateAngle = 0;
    QPointF m_translate;
    QPointF m_lastMousePos;
    QList<Annotation> m_annotations;
    bool m_isCreating = false;
    bool m_isMovingBox = false;
    bool m_isPanning = false;
    bool m_isResizingBox = false;
    ResizeHandle m_resizeHandle = NoHandle;
    QRectF m_tempRect;
    int m_selectedIndex = -1;
    int m_currentClassId = 0;
    QString m_currentClassName;
    QList<Annotation> m_gestureBefore;
    int m_gestureSelectionBefore = -1;
    QUndoStack m_undoStack;
    bool m_replayingSnapshot = false;
};

#endif
