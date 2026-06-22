#include "yoloutils.h"
#include <QFile>
#include <QTextStream>
#include <QFileInfo>
#include <QDir>

bool YoloUtils::saveAnnotations(const QString &saveDir,
                                const QString &imgPath,
                                const QList<Annotation> &annotations,
                                const QSize &imgSize)
{
    if (imgSize.isEmpty()) return false;

    QFileInfo imgInfo(imgPath);
    QString txtPath;

    // 如果没有指定独立的保存目录，则默认和图片放在一起
    if (saveDir.isEmpty()) {
        txtPath = imgInfo.absolutePath() + "/" + imgInfo.completeBaseName() + ".txt";
    } else {
        txtPath = QDir(saveDir).filePath(imgInfo.completeBaseName() + ".txt");
    }

    QFile file(txtPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) return false;

    QTextStream out(&file);
    qreal w = imgSize.width();
    qreal h = imgSize.height();

    for (const auto &ann : annotations) {
        const QRectF imageBounds(-w / 2.0, -h / 2.0, w, h);
        QRectF rect = ann.rect.normalized().intersected(imageBounds);
        if (rect.width() < 1.0 || rect.height() < 1.0 || ann.classId < 0) continue;
        qreal x = (rect.center().x() + w/2) / w;
        qreal y = (rect.center().y() + h/2) / h;
        qreal width = rect.width() / w;
        qreal height = rect.height() / h;

        out << ann.classId << " "
            << x << " " << y << " "
            << width << " " << height << "\n";
    }

    file.close();
    return true;
}

// 加载YOLO标注
QList<Annotation> YoloUtils::loadAnnotations(const QString &saveDir,
                                             const QString &imgPath,
                                             const QSize &imgSize,
                                             const QStringList &classList)
{
    QList<Annotation> anns;

    QFileInfo imgInfo(imgPath);
    QString txtPath;

    if (saveDir.isEmpty()) {
        txtPath = imgInfo.absolutePath() + "/" + imgInfo.completeBaseName() + ".txt";
    } else {
        txtPath = QDir(saveDir).filePath(imgInfo.completeBaseName() + ".txt");
    }

    QFile file(txtPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return anns;

    QTextStream in(&file);
    qreal w = imgSize.width();
    qreal h = imgSize.height();

    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;
        QStringList parts = line.split(" ", Qt::SkipEmptyParts);
        if (parts.size() != 5) continue;

        bool idOk = false, xOk = false, yOk = false, widthOk = false, heightOk = false;
        int classId = parts[0].toInt(&idOk);
        qreal xNorm = parts[1].toDouble(&xOk);
        qreal yNorm = parts[2].toDouble(&yOk);
        qreal widthNorm = parts[3].toDouble(&widthOk);
        qreal heightNorm = parts[4].toDouble(&heightOk);
        if (!idOk || !xOk || !yOk || !widthOk || !heightOk || classId < 0 ||
            xNorm < 0 || xNorm > 1 || yNorm < 0 || yNorm > 1 ||
            widthNorm <= 0 || widthNorm > 1 || heightNorm <= 0 || heightNorm > 1) continue;
        qreal x = xNorm * w - w/2;
        qreal y = yNorm * h - h/2;
        qreal width = widthNorm * w;
        qreal height = heightNorm * h;

        Annotation ann;
        ann.classId = classId;
        ann.className = classList.value(classId, "unknown");
        ann.rect = QRectF(x - width/2, y - height/2, width, height);
        ann.isSelected = false;
        anns.append(ann);
    }

    file.close();
    return anns;
}
