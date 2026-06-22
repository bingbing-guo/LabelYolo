#ifndef ONNXINFERENCEENGINE_H
#define ONNXINFERENCEENGINE_H

#include <QImage>
#include <QList>
#include <QString>
#include <QStringList>

struct DetectionResult {
    QRectF rect;
    int classId = -1;
    QString className;
    float confidence = 0.0f;
};

class OnnxInferenceEngine
{
public:
    OnnxInferenceEngine();
    ~OnnxInferenceEngine();
    bool isAvailable() const;
    bool load(const QString &modelPath, QString *error);
    QList<DetectionResult> infer(const QImage &image, float confidence, float iou, QString *error);
    void setClassNames(const QStringList &names) { m_classNames = names; }

private:
    class Impl;
    Impl *m_impl;
    QStringList m_classNames;
};

#endif
