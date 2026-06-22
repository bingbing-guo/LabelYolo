#ifndef YOLOUTILS_H
#define YOLOUTILS_H

#include <QString>
#include <QList>
#include <QSize>
#include "canvaswidget.h"
#include "Annotation.h"

class YoloUtils
{
public:
    // 新增 saveDir 参数
    static bool saveAnnotations(const QString &saveDir, const QString &imgPath,
                                const QList<Annotation> &annotations, const QSize &imgSize);

    // 新增 saveDir 参数
    static QList<Annotation> loadAnnotations(const QString &saveDir, const QString &imgPath,
                                             const QSize &imgSize, const QStringList &classList);
};

#endif // YOLOUTILS_H