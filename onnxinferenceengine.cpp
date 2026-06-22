#include "onnxinferenceengine.h"

#include <QFileInfo>
#include <QColor>
#include <QPainter>
#include <QPen>
#include <QRegularExpression>
#include <QtMath>
#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef YOLO_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace {
float overlap(const QRectF &a, const QRectF &b)
{
    const qreal intersection = a.intersected(b).width() * a.intersected(b).height();
    const qreal areaA = a.width() * a.height();
    const qreal areaB = b.width() * b.height();
    return float(intersection / qMax<qreal>(areaA + areaB - intersection, 1.0));
}

QList<DetectionResult> nms(QList<DetectionResult> detections, float threshold)
{
    std::sort(detections.begin(), detections.end(), [](const auto &a, const auto &b) { return a.confidence > b.confidence; });
    QList<DetectionResult> kept;
    while (!detections.isEmpty()) {
        DetectionResult best = detections.takeFirst();
        kept.append(best);
        for (int i = detections.size() - 1; i >= 0; --i)
            if (detections[i].classId == best.classId && overlap(detections[i].rect, best.rect) > threshold)
                detections.removeAt(i);
    }
    return kept;
}
}

class OnnxInferenceEngine::Impl
{
public:
#ifdef YOLO_WITH_ONNXRUNTIME
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "YOLOAnnotator"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string inputName;
    std::string outputName;
    int64_t inputWidth = 640;
    int64_t inputHeight = 640;
    bool endToEnd = false;
    QStringList modelClassNames;
#endif
};

OnnxInferenceEngine::OnnxInferenceEngine() : m_impl(new Impl) {}
OnnxInferenceEngine::~OnnxInferenceEngine() { delete m_impl; }

bool OnnxInferenceEngine::isAvailable() const
{
#ifdef YOLO_WITH_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

bool OnnxInferenceEngine::load(const QString &modelPath, QString *error)
{
#ifndef YOLO_WITH_ONNXRUNTIME
    Q_UNUSED(modelPath)
    if (error) *error = QStringLiteral("当前构建未启用 ONNX Runtime。请配置 ONNXRUNTIME_ROOT 后重新运行 CMake。");
    return false;
#else
    try {
        if (!QFileInfo::exists(modelPath)) throw std::runtime_error("model file does not exist");
        m_impl->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef Q_OS_WIN
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, modelPath.toStdWString().c_str(), m_impl->options);
#else
        m_impl->session = std::make_unique<Ort::Session>(m_impl->env, modelPath.toUtf8().constData(), m_impl->options);
#endif
        Ort::AllocatorWithDefaultOptions allocator;
        auto input = m_impl->session->GetInputNameAllocated(0, allocator);
        auto output = m_impl->session->GetOutputNameAllocated(0, allocator);
        m_impl->inputName = input.get(); m_impl->outputName = output.get();
        m_impl->endToEnd = false;
        m_impl->modelClassNames.clear();
        const auto metadata = m_impl->session->GetModelMetadata();
        if (auto endToEnd = metadata.LookupCustomMetadataMapAllocated("end2end", allocator)) {
            const QString value = QString::fromUtf8(endToEnd.get()).trimmed();
            m_impl->endToEnd = value.compare(QStringLiteral("true"), Qt::CaseInsensitive) == 0 || value == QStringLiteral("1");
        }
        if (auto names = metadata.LookupCustomMetadataMapAllocated("names", allocator)) {
            const QString value = QString::fromUtf8(names.get());
            static const QRegularExpression entryPattern(QStringLiteral(R"((\d+)\s*:\s*['"]([^'"]+)['"])"));
            auto match = entryPattern.globalMatch(value);
            while (match.hasNext()) {
                const auto entry = match.next();
                const int classId = entry.captured(1).toInt();
                while (m_impl->modelClassNames.size() <= classId) m_impl->modelClassNames.append(QString());
                m_impl->modelClassNames[classId] = entry.captured(2);
            }
        }
        const auto shape = m_impl->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4) { if (shape[2] > 0) m_impl->inputHeight = shape[2]; if (shape[3] > 0) m_impl->inputWidth = shape[3]; }
        return true;
    } catch (const std::exception &exception) {
        if (error) *error = QString::fromUtf8(exception.what());
        return false;
    }
#endif
}

QList<DetectionResult> OnnxInferenceEngine::infer(const QImage &source, float confidence, float iou, QString *error)
{
#ifndef YOLO_WITH_ONNXRUNTIME
    Q_UNUSED(source) Q_UNUSED(confidence) Q_UNUSED(iou)
    if (error) *error = QStringLiteral("ONNX Runtime 未启用");
    return {};
#else
    if (!m_impl->session || source.isNull()) { if (error) *error = QStringLiteral("模型未加载或图片无效"); return {}; }
    try {
        const int width = int(m_impl->inputWidth), height = int(m_impl->inputHeight);
        const float scale = qMin(float(width) / source.width(), float(height) / source.height());
        const int resizedWidth = qRound(source.width() * scale), resizedHeight = qRound(source.height() * scale);
        const int padX = (width - resizedWidth) / 2, padY = (height - resizedHeight) / 2;
        QImage canvas(width, height, QImage::Format_RGB888); canvas.fill(QColor(114,114,114));
        QPainter painter(&canvas); painter.drawImage(QRect(padX,padY,resizedWidth,resizedHeight), source.convertToFormat(QImage::Format_RGB888)); painter.end();
        std::vector<float> input(3 * width * height);
        for (int y=0; y<height; ++y) { const uchar *row=canvas.constScanLine(y); for(int x=0;x<width;++x) for(int c=0;c<3;++c) input[c*width*height+y*width+x]=row[x*3+c]/255.0f; }
        const std::array<int64_t,4> shape{1,3,height,width};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto tensor=Ort::Value::CreateTensor<float>(memory,input.data(),input.size(),shape.data(),shape.size());
        const char *inputs[]{m_impl->inputName.c_str()}, *outputs[]{m_impl->outputName.c_str()};
        auto result=m_impl->session->Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
        const auto outShape=result[0].GetTensorTypeAndShapeInfo().GetShape(); const float *data=result[0].GetTensorData<float>();
        if(outShape.size()!=3) throw std::runtime_error("unsupported YOLO output shape");
        const bool channelsFirst=outShape[1] < outShape[2]; const int64_t rows=channelsFirst?outShape[2]:outShape[1]; const int64_t cols=channelsFirst?outShape[1]:outShape[2];
        QList<DetectionResult> detections;
        auto at=[&](int64_t row,int64_t col){return channelsFirst?data[col*rows+row]:data[row*cols+col];};
        auto className = [&](int classId) {
            const QString modelName = m_impl->modelClassNames.value(classId);
            if (!modelName.isEmpty()) return modelName;
            const QString configuredName = m_classNames.value(classId);
            return configuredName.isEmpty() ? QString::number(classId) : configuredName;
        };

        // YOLO26 end-to-end exports return [batch, detections, 6], where each row is
        // [x1, y1, x2, y2, confidence, class_id]. These coordinates are already xyxy.
        const bool sixColumnEndToEnd = !channelsFirst && cols == 6 && rows <= 1000;
        if (m_impl->endToEnd || sixColumnEndToEnd) {
            for (int64_t row = 0; row < rows; ++row) {
                const float score = at(row, 4);
                const float rawClassId = at(row, 5);
                const int classId = qRound(rawClassId);
                if (!std::isfinite(score) || score < confidence || classId < 0
                    || qAbs(rawClassId - classId) > 0.01f) continue;

                const float x1 = (at(row, 0) - padX) / scale;
                const float y1 = (at(row, 1) - padY) / scale;
                const float x2 = (at(row, 2) - padX) / scale;
                const float y2 = (at(row, 3) - padY) / scale;
                QRectF box(QPointF(x1, y1), QPointF(x2, y2));
                box = box.normalized().intersected(QRectF(0, 0, source.width(), source.height()));
                if (box.width() <= 0 || box.height() <= 0) continue;

                DetectionResult detection;
                detection.rect = box;
                detection.classId = classId;
                detection.className = className(classId);
                detection.confidence = score;
                detections.append(detection);
            }
            return nms(detections, iou);
        }

        if (cols < 5) throw std::runtime_error("unsupported YOLO output columns");
        for(int64_t row=0;row<rows;++row){int best=-1;float score=0;for(int64_t c=4;c<cols;++c)if(at(row,c)>score){score=at(row,c);best=int(c-4);}if(score<confidence||best<0)continue;
            const float cx=at(row,0),cy=at(row,1),w=at(row,2),h=at(row,3); QRectF box((cx-w/2-padX)/scale,(cy-h/2-padY)/scale,w/scale,h/scale); box=box.intersected(QRectF(0,0,source.width(),source.height()));
            DetectionResult d;d.rect=box;d.classId=best;d.className=className(best);d.confidence=score;detections.append(d);}
        return nms(detections,iou);
    } catch(const std::exception &exception){if(error)*error=QString::fromUtf8(exception.what());return {};}
#endif
}
