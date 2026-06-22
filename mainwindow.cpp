#include "mainwindow.h"
#include "quantdialog.h"
#include "yoloutils.h"

#include <QApplication>
#include <QAction>
#include <QActionGroup>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDockWidget>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontDatabase>
#include <QFormLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QInputDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenuBar>
#include <QMessageBox>
#include <QProgressBar>
#include <QPlainTextEdit>
#include <QPixmap>
#include <QRegularExpression>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QSqlError>
#include <QSqlQuery>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTableWidget>
#include <QTextStream>
#include <QToolBar>
#include <QVBoxLayout>
#include <algorithm>
#include <utility>

namespace {
QString cleanTerminalText(const QByteArray &data)
{
    QString text = QString::fromUtf8(data);
    static const QRegularExpression ansiSequence(
        QStringLiteral("\\x1B(?:\\[[0-?]*[ -/]*[@-~]|\\][^\\x07]*(?:\\x07|\\x1B\\\\))"));
    text.remove(ansiSequence);
    text.remove(QChar(0x1b));
    text.remove(QChar('\b'));
    return text;
}

QString discoverRunsDirectory()
{
    QStringList roots{QDir::currentPath(), QCoreApplication::applicationDirPath()};
    QDir parent(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 6 && parent.cdUp(); ++depth) roots.append(parent.absolutePath());
    for (const QString &root : std::as_const(roots)) {
        const QDir runs(QDir(root).filePath("runs"));
        if (runs.exists()) return runs.absolutePath();
    }
    return {};
}
}

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    initUI();
    initTrainingUI();
    m_quantizationPage = new QuantDialog(m_pages);
    m_pages->addWidget(m_quantizationPage);
    initDatabase();
    loadSettings();
    showAnnotationPage();
}

MainWindow::~MainWindow()
{
    saveSettings();
    if (m_trainProcess && m_trainProcess->state() != QProcess::NotRunning) m_trainProcess->kill();
    if (m_trainingDb.isOpen()) m_trainingDb.close();
}

void MainWindow::initUI()
{
    setWindowTitle(tr("YOLO 标注、训练与推理工具"));
    resize(1400, 900);
    m_pages = new QStackedWidget(this);
    setCentralWidget(m_pages);
    m_annotationPage = new QWidget(m_pages);
    auto *annotationPageLayout = new QVBoxLayout(m_annotationPage);
    annotationPageLayout->setContentsMargins(6, 6, 6, 6);

    m_canvas = new CanvasWidget(m_annotationPage);

    auto *fileMenu = menuBar()->addMenu(tr("文件"));
    fileMenu->addAction(tr("打开图片目录"), this, &MainWindow::openFolder, QKeySequence::Open);
    fileMenu->addAction(tr("选择标签目录"), this, &MainWindow::chooseSaveDir);
    fileMenu->addAction(tr("保存标注"), this, &MainWindow::saveCurrentAnnotation, QKeySequence::Save);
    fileMenu->addSeparator();
    fileMenu->addAction(tr("退出"), this, &QWidget::close, QKeySequence::Quit);

    auto *editMenu = menuBar()->addMenu(tr("编辑"));
    QAction *undoAction = m_canvas->undoStack()->createUndoAction(this, tr("撤销"));
    auto *redoAction = new QAction(tr("重做（清空全部框）"), this);
    undoAction->setShortcut(QKeySequence::Undo);
    redoAction->setShortcut(QKeySequence::Redo);
    editMenu->addAction(undoAction);
    editMenu->addAction(redoAction);
    connect(redoAction, &QAction::triggered, this, &MainWindow::redoCurrentAnnotation);
    redoAction->setEnabled(false);
    connect(m_canvas, &CanvasWidget::annotationsChanged, this, [this, redoAction]() {
        redoAction->setEnabled(!m_canvas->getAnnotations().isEmpty());
    });
    editMenu->addAction(tr("删除选中框"), m_canvas, &CanvasWidget::deleteSelectedAnnotation, QKeySequence::Delete);

    auto *pageMenu = menuBar()->addMenu(tr("工作区"));
    auto *pageActions = new QActionGroup(this);
    pageActions->setExclusive(true);
    m_annotationPageAction = pageMenu->addAction(tr("图片标注"));
    m_trainingPageAction = pageMenu->addAction(tr("模型训练"));
    m_quantizationPageAction = pageMenu->addAction(tr("量化与推理"));
    for (QAction *action : {m_annotationPageAction, m_trainingPageAction, m_quantizationPageAction}) {
        action->setCheckable(true);
        pageActions->addAction(action);
    }
    connect(m_annotationPageAction, &QAction::triggered, this, &MainWindow::showAnnotationPage);
    connect(m_trainingPageAction, &QAction::triggered, this, &MainWindow::showTrainingPage);
    connect(m_quantizationPageAction, &QAction::triggered, this, &MainWindow::showQuantizationPage);

    auto *operations = new QHBoxLayout;
    auto addOperation = [operations](const QString &text, auto receiver, auto slot) {
        auto *button = new QPushButton(text);
        QObject::connect(button, &QPushButton::clicked, receiver, slot);
        operations->addWidget(button);
    };
    addOperation(tr("添加类别"), this, &MainWindow::addNewClass);
    addOperation(tr("删除类别"), this, &MainWindow::deleteCurrentClass);
    auto *undoButton = new QPushButton(tr("撤销"), m_annotationPage);
    auto *redoButton = new QPushButton(tr("重做（清空全部框）"), m_annotationPage);
    undoButton->setEnabled(m_canvas->undoStack()->canUndo());
    redoButton->setEnabled(!m_canvas->getAnnotations().isEmpty());
    connect(undoButton, &QPushButton::clicked, m_canvas->undoStack(), &QUndoStack::undo);
    connect(redoButton, &QPushButton::clicked, this, &MainWindow::redoCurrentAnnotation);
    connect(m_canvas->undoStack(), &QUndoStack::canUndoChanged, undoButton, &QPushButton::setEnabled);
    connect(m_canvas, &CanvasWidget::annotationsChanged, this, [this, redoButton]() {
        redoButton->setEnabled(!m_canvas->getAnnotations().isEmpty());
    });
    operations->addWidget(undoButton);
    operations->addWidget(redoButton);
    m_classCombo = new QComboBox(m_annotationPage);
    m_classCombo->setMinimumWidth(150);
    operations->addWidget(new QLabel(tr("当前类别")));
    operations->addWidget(m_classCombo);
    operations->addStretch();
    addOperation(tr("上一张"), this, &MainWindow::prevImage);
    addOperation(tr("下一张"), this, &MainWindow::nextImage);
    addOperation(tr("旋转 90°"), m_canvas, &CanvasWidget::rotateClockwise);
    annotationPageLayout->addLayout(operations);

    auto *contentSplitter = new QSplitter(Qt::Horizontal, m_annotationPage);
    auto *annotationPanel = new QWidget(contentSplitter);
    auto *annotationLayout = new QVBoxLayout(annotationPanel);
    annotationLayout->setContentsMargins(4, 4, 4, 4);
    auto *annotationTitle = new QLabel(tr("标注详情"), annotationPanel);
    annotationTitle->setStyleSheet("font-weight: bold");
    m_datasetSummary = new QLabel(tr("尚未打开数据集"), annotationPanel);
    m_datasetSummary->setWordWrap(true);
    m_annotationList = new QListWidget(annotationPanel);
    annotationLayout->addWidget(annotationTitle);
    annotationLayout->addWidget(m_datasetSummary);
    annotationLayout->addWidget(m_annotationList, 1);

    auto *imagePanel = new QWidget(contentSplitter);
    auto *imageLayout = new QVBoxLayout(imagePanel);
    imageLayout->setContentsMargins(4, 4, 4, 4);
    auto *imageTitle = new QLabel(tr("图片列表"), imagePanel);
    imageTitle->setStyleSheet("font-weight: bold");
    m_imageList = new QListWidget(imagePanel);
    imageLayout->addWidget(imageTitle);
    imageLayout->addWidget(m_imageList, 1);

    contentSplitter->addWidget(annotationPanel);
    contentSplitter->addWidget(m_canvas);
    contentSplitter->addWidget(imagePanel);
    contentSplitter->setStretchFactor(0, 0);
    contentSplitter->setStretchFactor(1, 1);
    contentSplitter->setStretchFactor(2, 0);
    contentSplitter->setSizes({260, 900, 240});
    annotationPageLayout->addWidget(contentSplitter, 1);
    m_pages->addWidget(m_annotationPage);

    m_imageStatus = new QLabel(tr("无图片"), this);
    m_zoomStatus = new QLabel(tr("缩放 100%"), this);
    m_annotationStatus = new QLabel(tr("标注 0"), this);
    statusBar()->addPermanentWidget(m_imageStatus, 1);
    statusBar()->addPermanentWidget(m_zoomStatus);
    statusBar()->addPermanentWidget(m_annotationStatus);

    connect(m_imageList, &QListWidget::currentRowChanged, this, &MainWindow::onImageSelected);
    connect(m_classCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onClassSelected);
    connect(m_canvas, &CanvasWidget::annotationsChanged, this, &MainWindow::refreshAnnotationList);
    connect(m_canvas, &CanvasWidget::selectionChanged, this, [this](int index) {
        if (m_annotationList->currentRow() != index) m_annotationList->setCurrentRow(index);
        const auto annotations = m_canvas->getAnnotations();
        if (index >= 0 && index < annotations.size()) {
            const int comboIndex = m_classCombo->findData(annotations[index].classId);
            if (comboIndex >= 0) m_classCombo->setCurrentIndex(comboIndex);
        }
    });
    connect(m_annotationList, &QListWidget::currentRowChanged, m_canvas, &CanvasWidget::selectAnnotation);
    connect(m_canvas, &CanvasWidget::viewChanged, this, [this](qreal scale) {
        m_zoomStatus->setText(tr("缩放 %1%").arg(qRound(scale * 100)));
    });
}

void MainWindow::showAnnotationPage()
{
    m_pages->setCurrentWidget(m_annotationPage);
    m_annotationPageAction->setChecked(true);
    m_imageStatus->show();
    m_zoomStatus->show();
    m_annotationStatus->show();
    statusBar()->showMessage(tr("图片标注工作区"), 1500);
}

void MainWindow::showTrainingPage()
{
    m_pages->setCurrentWidget(m_trainingPage);
    m_trainingPageAction->setChecked(true);
    m_imageStatus->hide();
    m_zoomStatus->hide();
    m_annotationStatus->hide();
    statusBar()->showMessage(tr("模型训练工作区"), 1500);
}

void MainWindow::showQuantizationPage()
{
    m_pages->setCurrentWidget(m_quantizationPage);
    m_quantizationPageAction->setChecked(true);
    m_imageStatus->hide();
    m_zoomStatus->hide();
    m_annotationStatus->hide();
    statusBar()->showMessage(tr("模型量化与推理工作区"), 1500);
}

void MainWindow::loadSettings()
{
    QSettings settings("MyAnnotator", "YoloTools");
    m_classList = settings.value("classes", QStringList{QStringLiteral("motorbike")}).toStringList();
    if (m_classList.isEmpty()) m_classList << QStringLiteral("motorbike");
    rebuildClassCombo();
    m_saveDir = settings.value("annotationDir").toString();
    m_pythonEdit->setText(settings.value("python", QStringLiteral("python")).toString());
    m_ultralyticsEdit->setText(settings.value("ultralyticsRoot").toString());
    const QString discoveredRuns = discoverRunsDirectory();
    m_outputEdit->setText(settings.value("outputDir",
        discoveredRuns.isEmpty() ? QDir::home().filePath("YOLORuns") : discoveredRuns).toString());
    m_trainDirEdit->setText(settings.value("trainDir").toString());
    m_valDirEdit->setText(settings.value("valDir").toString());
    const QString defaultModel = QDir(QCoreApplication::applicationDirPath()).filePath("yolo26n.pt");
    QString configuredModel = settings.value("model").toString();
    if (configuredModel.isEmpty() || configuredModel == QStringLiteral("yolov8n.pt"))
        configuredModel = defaultModel;
    m_modelEdit->setText(configuredModel);
    m_deviceEdit->setText(settings.value("device").toString());
    loadLatestTrainingResultImages();
}

void MainWindow::saveSettings()
{
    QSettings settings("MyAnnotator", "YoloTools");
    settings.setValue("classes", m_classList);
    settings.setValue("annotationDir", m_saveDir);
    settings.setValue("python", m_pythonEdit->text());
    settings.setValue("ultralyticsRoot", m_ultralyticsEdit->text());
    settings.setValue("outputDir", m_outputEdit->text());
    settings.setValue("trainDir", m_trainDirEdit->text());
    settings.setValue("valDir", m_valDirEdit->text());
    settings.setValue("model", m_modelEdit->text());
    settings.setValue("device", m_deviceEdit->text());
}

void MainWindow::rebuildClassCombo(int preferredId)
{
    if (preferredId < 0 && m_classCombo->currentIndex() >= 0) preferredId = m_classCombo->currentData().toInt();
    m_classCombo->blockSignals(true);
    m_classCombo->clear();
    for (int id = 0; id < m_classList.size(); ++id)
        if (!m_classList[id].isEmpty()) m_classCombo->addItem(m_classList[id], id);
    int index = m_classCombo->findData(preferredId);
    if (index < 0 && m_classCombo->count()) index = 0;
    m_classCombo->setCurrentIndex(index);
    m_classCombo->blockSignals(false);
    onClassSelected(index);
}

void MainWindow::openFolder()
{
    const QString folder = QFileDialog::getExistingDirectory(this, tr("选择图片目录"));
    if (folder.isEmpty()) return;
    QDir dir(folder);
    const QStringList filters{"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"};
    const QStringList names = dir.entryList(filters, QDir::Files, QDir::Name);
    m_imagePaths.clear();
    for (const QString &name : names) m_imagePaths << dir.absoluteFilePath(name);
    m_imageList->clear();
    for (const QString &path : m_imagePaths) m_imageList->addItem(QFileInfo(path).fileName());
    m_currentImageIndex = -1;
    if (!m_imagePaths.isEmpty()) m_imageList->setCurrentRow(0);
    updateDatasetSummary();
}

void MainWindow::chooseSaveDir()
{
    const QString dir = QFileDialog::getExistingDirectory(this, tr("选择标签保存目录"), m_saveDir);
    if (!dir.isEmpty()) { m_saveDir = dir; saveSettings(); updateDatasetSummary(); }
}

void MainWindow::onImageSelected(int index)
{
    if (index < 0 || index >= m_imagePaths.size()) return;
    if (m_currentImageIndex >= 0 && m_currentImageIndex != index) saveCurrentAnnotation();
    const QImage image(m_imagePaths[index]);
    if (image.isNull()) { QMessageBox::warning(this, tr("图片错误"), tr("无法加载：%1").arg(m_imagePaths[index])); return; }
    m_currentImageIndex = index;
    m_canvas->setImage(image);
    m_canvas->loadAnnotations(YoloUtils::loadAnnotations(m_saveDir, m_imagePaths[index], image.size(), m_classList));
    m_imageStatus->setText(tr("%1 / %2  %3").arg(index + 1).arg(m_imagePaths.size()).arg(QFileInfo(m_imagePaths[index]).fileName()));
    refreshAnnotationList();
}

void MainWindow::saveCurrentAnnotation()
{
    if (m_currentImageIndex < 0 || m_currentImageIndex >= m_imagePaths.size()) return;
    if (!YoloUtils::saveAnnotations(m_saveDir, m_imagePaths[m_currentImageIndex],
                                    m_canvas->getAnnotations(), m_canvas->getImageSize()))
        QMessageBox::warning(this, tr("保存失败"), tr("无法写入标签文件，请检查目录权限。"));
    else statusBar()->showMessage(tr("标注已保存"), 2000);
    updateDatasetSummary();
}

void MainWindow::addNewClass()
{
    bool ok = false;
    const QString name = QInputDialog::getText(this, tr("添加类别"), tr("类别名称"), QLineEdit::Normal, {}, &ok).trimmed();
    if (!ok || name.isEmpty()) return;
    if (m_classList.contains(name)) { QMessageBox::information(this, tr("类别已存在"), tr("该类别已经存在。")); return; }
    m_classList.append(name);
    rebuildClassCombo(m_classList.size() - 1);
    saveSettings();
}

void MainWindow::deleteCurrentClass()
{
    if (m_classCombo->currentIndex() < 0) return;
    const int id = m_classCombo->currentData().toInt();
    for (const Annotation &annotation : m_canvas->getAnnotations()) {
        if (annotation.classId == id) {
            QMessageBox::warning(this, tr("类别正在使用"), tr("当前图片仍有该类别标注，请先将这些框改为其他类别。"));
            return;
        }
    }
    if (QMessageBox::question(this, tr("删除类别"), tr("删除类别“%1”？类别 ID %2 将保留为空位，不会重排其他 ID。")
                              .arg(m_classList.value(id)).arg(id)) != QMessageBox::Yes) return;
    m_classList[id].clear();
    if (std::all_of(m_classList.cbegin(), m_classList.cend(), [](const QString &v){ return v.isEmpty(); })) m_classList.append("object");
    rebuildClassCombo();
    saveSettings();
}

void MainWindow::redoCurrentAnnotation()
{
    const int count = m_canvas->getAnnotations().size();
    if (count == 0) return;
    if (QMessageBox::question(this, tr("重新标注"),
                              tr("将清空当前图片中的 %1 个标注框，之后可使用“撤销”恢复。是否继续？").arg(count))
        != QMessageBox::Yes) return;
    m_canvas->clearAnnotations();
    statusBar()->showMessage(tr("已清空当前图片标注，可重新绘制；撤销可恢复"), 3000);
}

void MainWindow::onClassSelected(int comboIndex)
{
    if (comboIndex < 0) return;
    const int id = m_classCombo->itemData(comboIndex).toInt();
    const QString name = m_classCombo->itemText(comboIndex);
    m_canvas->addAnnotationClass(name, id);
    if (m_canvas->selectedIndex() >= 0) m_canvas->changeSelectedClass(name, id);
}

void MainWindow::refreshAnnotationList()
{
    const auto annotations = m_canvas->getAnnotations();
    m_annotationList->blockSignals(true);
    m_annotationList->clear();
    for (int i = 0; i < annotations.size(); ++i) {
        const QRectF r = annotations[i].rect;
        m_annotationList->addItem(tr("%1. %2  x:%3 y:%4 w:%5 h:%6").arg(i + 1).arg(annotations[i].className)
                                  .arg(qRound(r.x())).arg(qRound(r.y())).arg(qRound(r.width())).arg(qRound(r.height())));
    }
    m_annotationList->setCurrentRow(m_canvas->selectedIndex());
    m_annotationList->blockSignals(false);
    m_annotationStatus->setText(tr("标注 %1").arg(annotations.size()));
}

void MainWindow::updateDatasetSummary()
{
    int annotated = 0;
    for (const QString &image : m_imagePaths) {
        const QFileInfo info(image);
        const QString label = m_saveDir.isEmpty() ? info.dir().filePath(info.completeBaseName() + ".txt")
                                                   : QDir(m_saveDir).filePath(info.completeBaseName() + ".txt");
        if (QFileInfo::exists(label)) ++annotated;
    }
    m_datasetSummary->setText(tr("图片 %1｜已标注 %2｜未标注 %3\n标签目录：%4")
                              .arg(m_imagePaths.size()).arg(annotated).arg(m_imagePaths.size() - annotated)
                              .arg(m_saveDir.isEmpty() ? tr("图片同目录") : m_saveDir));
}

void MainWindow::prevImage() { if (m_currentImageIndex > 0) m_imageList->setCurrentRow(m_currentImageIndex - 1); }
void MainWindow::nextImage() { if (m_currentImageIndex + 1 < m_imagePaths.size()) m_imageList->setCurrentRow(m_currentImageIndex + 1); }

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Delete) m_canvas->deleteSelectedAnnotation();
    else if (event->key() == Qt::Key_R) m_canvas->rotateClockwise();
    else if (event->key() == Qt::Key_A || event->key() == Qt::Key_Up) prevImage();
    else if (event->key() == Qt::Key_Down || (event->key() == Qt::Key_S && !(event->modifiers() & Qt::ControlModifier))) nextImage();
    else QMainWindow::keyPressEvent(event);
}

void MainWindow::initTrainingUI()
{
    auto *panel = new QWidget(m_pages);
    m_trainingPage = panel;
    auto *layout = new QVBoxLayout(panel);
    auto *configurationLayout = new QHBoxLayout;
    auto *paths = new QGroupBox(tr("数据集与运行环境"), panel);
    auto *form = new QFormLayout(paths);
    auto addPathRow = [form](const QString &label, QLineEdit *&edit, const QObject *receiver, void (MainWindow::*slot)()) {
        auto *row = new QWidget; auto *rowLayout = new QHBoxLayout(row); rowLayout->setContentsMargins(0,0,0,0);
        edit = new QLineEdit; auto *button = new QPushButton(QObject::tr("浏览"));
        QObject::connect(button, &QPushButton::clicked, receiver, slot);
        rowLayout->addWidget(edit, 1); rowLayout->addWidget(button); form->addRow(label, row);
    };
    addPathRow(tr("训练图片目录"), m_trainDirEdit, this, &MainWindow::chooseTrainDir);
    addPathRow(tr("验证图片目录"), m_valDirEdit, this, &MainWindow::chooseValDir);
    addPathRow(tr("Python"), m_pythonEdit, this, &MainWindow::choosePython);
    addPathRow(tr("Ultralytics 源码（可空）"), m_ultralyticsEdit, this, &MainWindow::chooseUltralyticsRoot);
    addPathRow(tr("输出目录"), m_outputEdit, this, &MainWindow::chooseOutputDir);
    configurationLayout->addWidget(paths, 1);

    auto *parameters = new QGroupBox(tr("训练参数"), panel);
    auto *params = new QFormLayout(parameters);
    m_modelEdit = new QLineEdit; m_deviceEdit = new QLineEdit;
    m_epochsSpin = new QSpinBox; m_epochsSpin->setRange(1, 10000); m_epochsSpin->setValue(100);
    m_batchSpin = new QSpinBox; m_batchSpin->setRange(1, 1024); m_batchSpin->setValue(16);
    m_imgSizeSpin = new QSpinBox; m_imgSizeSpin->setRange(32, 4096); m_imgSizeSpin->setSingleStep(32); m_imgSizeSpin->setValue(640);
    m_lrSpin = new QDoubleSpinBox; m_lrSpin->setRange(0.000001, 1.0); m_lrSpin->setDecimals(6); m_lrSpin->setValue(0.01);
    m_optimizerCombo = new QComboBox; m_optimizerCombo->addItems({"auto", "SGD", "Adam", "AdamW"});
    params->addRow(tr("模型"), m_modelEdit); params->addRow(tr("设备（空=自动）"), m_deviceEdit);
    params->addRow("Epochs", m_epochsSpin); params->addRow("Batch", m_batchSpin); params->addRow("Image size", m_imgSizeSpin);
    params->addRow("Learning rate", m_lrSpin); params->addRow(tr("优化器"), m_optimizerCombo);
    configurationLayout->addWidget(parameters, 1);
    layout->addLayout(configurationLayout);

    auto *actions = new QWidget(panel);
    auto *buttonLayout = new QHBoxLayout(actions); buttonLayout->setContentsMargins(0,0,0,0);
    m_trainBtn = new QPushButton(tr("开始训练")); m_stopBtn = new QPushButton(tr("停止训练")); m_stopBtn->setEnabled(false);
    m_trainProgress = new QProgressBar; m_trainProgress->setRange(0, 100);
    buttonLayout->addWidget(m_trainBtn); buttonLayout->addWidget(m_stopBtn);
    buttonLayout->addSpacing(12); buttonLayout->addWidget(new QLabel(tr("进度")));
    buttonLayout->addWidget(m_trainProgress, 1);
    layout->addWidget(actions);

    m_logOutput = new QPlainTextEdit; m_logOutput->setReadOnly(true); m_logOutput->setMinimumHeight(130);
    m_logOutput->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    m_logOutput->setLineWrapMode(QPlainTextEdit::NoWrap);
    m_logOutput->setMaximumHeight(150);
    layout->addWidget(m_logOutput);

    m_trainingSummaryTable = new QTableWidget(0, 3);
    m_trainingSummaryTable->setHorizontalHeaderLabels({tr("类型"), tr("名称"), tr("值 / 文件路径")});
    m_trainingSummaryTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_trainingSummaryTable->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    m_trainingSummaryTable->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    m_trainingSummaryTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_trainingSummaryTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_trainingSummaryTable->setMaximumHeight(210);
    layout->addWidget(m_trainingSummaryTable);

    auto *resultHeader = new QHBoxLayout;
    resultHeader->addWidget(new QLabel(tr("训练结果图")));
    m_trainingResultNameLabel = new QLabel(tr("暂无训练结果"));
    m_trainingResultNameLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    resultHeader->addWidget(m_trainingResultNameLabel, 1);
    m_trainingResultPrevBtn = new QPushButton(tr("上一张"));
    m_trainingResultPageLabel = new QLabel("0 / 0");
    m_trainingResultNextBtn = new QPushButton(tr("下一张"));
    auto *chooseResultDirBtn = new QPushButton(tr("选择结果目录"));
    resultHeader->addWidget(chooseResultDirBtn);
    resultHeader->addWidget(m_trainingResultPrevBtn);
    resultHeader->addWidget(m_trainingResultPageLabel);
    resultHeader->addWidget(m_trainingResultNextBtn);
    layout->addLayout(resultHeader);

    m_trainingResultImageLabel = new QLabel(tr("训练完成后将在此展示结果图片"));
    m_trainingResultImageLabel->setAlignment(Qt::AlignCenter);
    m_trainingResultImageLabel->setMinimumHeight(320);
    m_trainingResultImageLabel->setStyleSheet("QLabel { background: #202020; color: #bdbdbd; border: 1px solid #505050; }");
    layout->addWidget(m_trainingResultImageLabel, 1);

    connect(m_trainingResultPrevBtn, &QPushButton::clicked, this, [this]() {
        if (m_trainingResultIndex > 0) { --m_trainingResultIndex; updateTrainingResultImage(); }
    });
    connect(m_trainingResultNextBtn, &QPushButton::clicked, this, [this]() {
        if (m_trainingResultIndex + 1 < m_trainingResultImages.size()) { ++m_trainingResultIndex; updateTrainingResultImage(); }
    });
    connect(chooseResultDirBtn, &QPushButton::clicked, this, &MainWindow::chooseTrainingResultDir);

    m_pages->addWidget(m_trainingPage);
    m_trainProcess = new QProcess(this); m_trainProcess->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_trainBtn, &QPushButton::clicked, this, &MainWindow::startTraining);
    connect(m_stopBtn, &QPushButton::clicked, this, &MainWindow::stopTraining);
    connect(m_trainProcess, &QProcess::readyReadStandardOutput, this, &MainWindow::readProcessOutput);
    connect(m_trainProcess, QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished), this, &MainWindow::processFinished);
}

void MainWindow::chooseTrainDir() { const QString v=QFileDialog::getExistingDirectory(this,tr("训练图片目录"),m_trainDirEdit->text()); if(!v.isEmpty())m_trainDirEdit->setText(v); }
void MainWindow::chooseValDir() { const QString v=QFileDialog::getExistingDirectory(this,tr("验证图片目录"),m_valDirEdit->text()); if(!v.isEmpty())m_valDirEdit->setText(v); }
void MainWindow::choosePython() { const QString v=QFileDialog::getOpenFileName(this,tr("Python 解释器"),m_pythonEdit->text(),"Python (python.exe python3.exe);;All files (*)"); if(!v.isEmpty())m_pythonEdit->setText(v); }
void MainWindow::chooseUltralyticsRoot() { const QString v=QFileDialog::getExistingDirectory(this,tr("Ultralytics 源码目录"),m_ultralyticsEdit->text()); if(!v.isEmpty())m_ultralyticsEdit->setText(v); }
void MainWindow::chooseOutputDir() { const QString v=QFileDialog::getExistingDirectory(this,tr("训练输出目录"),m_outputEdit->text()); if(!v.isEmpty()){m_outputEdit->setText(v);loadLatestTrainingResultImages();} }
void MainWindow::chooseTrainingResultDir() { const QString start=m_trainingResultDirectory.isEmpty()?m_outputEdit->text():m_trainingResultDirectory; const QString v=QFileDialog::getExistingDirectory(this,tr("选择训练结果目录"),start); if(!v.isEmpty())loadTrainingResultImages(v); }

QString MainWindow::scriptPath(const QString &name) const { return QDir(QCoreApplication::applicationDirPath()).filePath(name); }

bool MainWindow::createDatasetYaml(QString *path, QString *error) const
{
    const QDir train(m_trainDirEdit->text()), val(m_valDirEdit->text());
    if (!train.exists() || !val.exists()) { *error = tr("训练或验证图片目录不存在。"); return false; }
    auto labelsFor = [](const QDir &images) {
        QString path = QDir::fromNativeSeparators(images.absolutePath());
        const int marker = path.lastIndexOf("/images/", -1, Qt::CaseInsensitive);
        if (marker >= 0) return QDir(path.left(marker) + "/labels/" + path.mid(marker + 8));
        if (path.endsWith("/images", Qt::CaseInsensitive)) return QDir(path.left(path.size() - 7) + "/labels");
        return QDir(images.filePath("labels"));
    };
    if (!labelsFor(train).exists() || !labelsFor(val).exists()) { *error = tr("未找到对应 labels 目录。支持 dataset/images + dataset/labels 结构。"); return false; }
    QDir output(m_outputEdit->text()); if (!output.exists() && !QDir().mkpath(output.absolutePath())) { *error=tr("无法创建输出目录。"); return false; }
    *path = output.filePath(QString("dataset_%1.yaml").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")));
    QFile file(*path); if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) { *error=tr("无法写入临时 data.yaml。"); return false; }
    QTextStream stream(&file);
    stream << "train: '" << QDir::fromNativeSeparators(train.absolutePath()).replace("'", "''") << "'\n";
    stream << "val: '" << QDir::fromNativeSeparators(val.absolutePath()).replace("'", "''") << "'\n";
    stream << "names:\n";
    for (int id=0; id<m_classList.size(); ++id) if(!m_classList[id].isEmpty()) {
        QString escapedName = m_classList[id];
        stream << "  " << id << ": '" << escapedName.replace("'", "''") << "'\n";
    }
    return true;
}

void MainWindow::startTraining()
{
    QString yaml, error;
    if (!createDatasetYaml(&yaml, &error)) { QMessageBox::warning(this,tr("数据集配置错误"),error); return; }
    const QString bridge = scriptPath("qt_train_bridge.py");
    if (!QFileInfo::exists(bridge)) { QMessageBox::warning(this,tr("缺少桥接脚本"),tr("未找到 %1，请将脚本部署到程序目录。").arg(bridge)); return; }
    saveSettings(); m_trainBuffer.clear(); m_logOutput->clear();
    m_trainingSummaryTable->setRowCount(0);
    m_trainingResultImages.clear(); m_trainingResultIndex = -1; updateTrainingResultImage();
    const QString runName = QString("train_%1").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
    QStringList args{"-u", bridge, "--data", yaml, "--model", m_modelEdit->text(), "--epochs", QString::number(m_epochsSpin->value()),
                     "--batch", QString::number(m_batchSpin->value()), "--imgsz", QString::number(m_imgSizeSpin->value()),
                     "--device", m_deviceEdit->text(), "--lr0", QString::number(m_lrSpin->value(),'g',8),
                     "--optimizer", m_optimizerCombo->currentText(), "--project", m_outputEdit->text(), "--name", runName};
    if (!m_ultralyticsEdit->text().isEmpty()) args << "--ultralytics-root" << m_ultralyticsEdit->text();
    QSqlQuery query(m_trainingDb); query.prepare("INSERT INTO training_tasks(started_at,status,data_yaml,model,output_dir) VALUES(?,?,?,?,?)");
    query.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate)); query.addBindValue("running"); query.addBindValue(yaml);
    query.addBindValue(m_modelEdit->text()); query.addBindValue(m_outputEdit->text()); query.exec(); m_trainingTaskId=query.lastInsertId().toLongLong();
    m_trainBtn->setEnabled(false); m_stopBtn->setEnabled(true); m_trainProgress->setValue(0);
    m_trainProcess->setWorkingDirectory(m_outputEdit->text()); m_trainProcess->start(m_pythonEdit->text().trimmed().isEmpty()?"python":m_pythonEdit->text(), args);
    if (!m_trainProcess->waitForStarted(3000)) processFinished(-1, QProcess::CrashExit);
}

void MainWindow::stopTraining() { if(m_trainProcess->state()!=QProcess::NotRunning){m_trainProcess->terminate(); if(!m_trainProcess->waitForFinished(3000))m_trainProcess->kill();} }
void MainWindow::readProcessOutput()
{
    m_trainBuffer += m_trainProcess->readAllStandardOutput();
    while (true) {
        const int newline = m_trainBuffer.indexOf('\n');
        const int carriageReturn = m_trainBuffer.indexOf('\r');
        int pos = -1;
        if (newline >= 0 && carriageReturn >= 0) pos = std::min(newline, carriageReturn);
        else pos = std::max(newline, carriageReturn);
        if (pos < 0) break;

        handleTrainingMessage(m_trainBuffer.left(pos));
        int consumed = pos + 1;
        if (m_trainBuffer.at(pos) == '\r' && consumed < m_trainBuffer.size()
            && m_trainBuffer.at(consumed) == '\n') ++consumed;
        m_trainBuffer.remove(0, consumed);
    }
}

void MainWindow::handleTrainingMessage(const QByteArray &line)
{
    const QString displayLine = cleanTerminalText(line);
    if(displayLine.trimmed().isEmpty())return;
    QByteArray jsonData = displayLine.toUtf8();
    QJsonParseError parse;
    QJsonDocument doc = QJsonDocument::fromJson(jsonData, &parse);
    if (parse.error != QJsonParseError::NoError || !doc.isObject()) {
        const int eventStart = jsonData.indexOf("{\"event\"");
        if (eventStart >= 0) {
            jsonData = jsonData.mid(eventStart);
            doc = QJsonDocument::fromJson(jsonData, &parse);
        }
    }
    if(parse.error!=QJsonParseError::NoError || !doc.isObject()){m_logOutput->appendPlainText(displayLine);return;}
    const QJsonObject object=doc.object(); const QString event=object.value("event").toString();
    if(event=="epoch") { const int epoch=object.value("epoch").toInt(); const QJsonObject metrics=object.value("metrics").toObject();
        m_trainProgress->setValue(qRound(epoch*100.0/m_epochsSpin->value()));
        QSqlQuery q(m_trainingDb); for(auto it=metrics.begin();it!=metrics.end();++it){q.prepare("INSERT INTO training_metrics(task_id,epoch,name,value) VALUES(?,?,?,?)");q.addBindValue(m_trainingTaskId);q.addBindValue(epoch);q.addBindValue(it.key());q.addBindValue(it.value().toDouble());q.exec();}
    } else if(event=="completed") { QSqlQuery q(m_trainingDb);q.prepare("UPDATE training_tasks SET status='completed',finished_at=?,best_model=?,last_model=? WHERE id=?");q.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));q.addBindValue(object.value("best").toString());q.addBindValue(object.value("last").toString());q.addBindValue(m_trainingTaskId);q.exec();loadTrainingResultImages(object.value("save_dir").toString());m_logOutput->appendPlainText(tr("训练完成，最佳模型：%1").arg(object.value("best").toString()));
    } else if(event=="error") m_logOutput->appendPlainText(tr("错误：%1").arg(object.value("message").toString()));
    else m_logOutput->appendPlainText(displayLine);
}

void MainWindow::loadTrainingResultImages(const QString &directory)
{
    m_trainingResultDirectory = QDir(directory).absolutePath();
    m_trainingResultImages.clear();
    m_trainingResultIndex = -1;
    const QDir resultDir(directory);
    if (resultDir.exists()) {
        const QFileInfoList files = resultDir.entryInfoList(
            {"*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"}, QDir::Files, QDir::Name);
        for (const QFileInfo &file : files) m_trainingResultImages.append(file.absoluteFilePath());
        if (!m_trainingResultImages.isEmpty()) m_trainingResultIndex = 0;
    }
    loadTrainingSummary(directory);
    updateTrainingResultImage();
}

void MainWindow::loadTrainingSummary(const QString &directory)
{
    m_trainingSummaryTable->setRowCount(0);
    auto addRow = [this](const QString &type, const QString &name, const QString &value) {
        const int row = m_trainingSummaryTable->rowCount();
        m_trainingSummaryTable->insertRow(row);
        m_trainingSummaryTable->setItem(row, 0, new QTableWidgetItem(type));
        m_trainingSummaryTable->setItem(row, 1, new QTableWidgetItem(name));
        m_trainingSummaryTable->setItem(row, 2, new QTableWidgetItem(value));
    };

    const QDir resultDir(directory);
    if (!resultDir.exists()) return;
    addRow(tr("训练"), tr("结果目录"), resultDir.absolutePath());
    for (const QString &modelName : {QStringLiteral("best.pt"), QStringLiteral("last.pt")}) {
        const QFileInfo model(resultDir.filePath("weights/" + modelName));
        addRow(tr("模型"), modelName, model.exists() ? model.absoluteFilePath() : tr("未生成"));
    }

    QFile csv(resultDir.filePath("results.csv"));
    if (!csv.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    QTextStream stream(&csv);
    const QStringList headers = stream.readLine().split(',');
    QString lastLine;
    while (!stream.atEnd()) {
        const QString line = stream.readLine().trimmed();
        if (!line.isEmpty()) lastLine = line;
    }
    const QStringList values = lastLine.split(',');
    const int count = std::min(headers.size(), values.size());
    for (int i = 0; i < count; ++i) {
        const QString name = headers.at(i).trimmed();
        if (name == "epoch" || name.contains("loss", Qt::CaseInsensitive)
            || name.startsWith("metrics/", Qt::CaseInsensitive)) {
            addRow(tr("指标"), name, values.at(i).trimmed());
        }
    }
}

void MainWindow::loadLatestTrainingResultImages()
{
    QDir outputDir(m_outputEdit->text());
    QFileInfoList runs = outputDir.entryInfoList(
        {"train_*"}, QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
    if (runs.isEmpty()) {
        const QString discoveredRuns = discoverRunsDirectory();
        if (!discoveredRuns.isEmpty()) {
            outputDir.setPath(discoveredRuns);
            runs = outputDir.entryInfoList(
                {"train_*"}, QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
        }
    }
    if (runs.isEmpty()) {
        m_trainingResultImages.clear();
        m_trainingResultIndex = -1;
        updateTrainingResultImage();
        return;
    }
    loadTrainingResultImages(runs.constFirst().absoluteFilePath());
}

void MainWindow::updateTrainingResultImage()
{
    const bool hasImage = m_trainingResultIndex >= 0
        && m_trainingResultIndex < m_trainingResultImages.size();
    m_trainingResultPrevBtn->setEnabled(hasImage && m_trainingResultIndex > 0);
    m_trainingResultNextBtn->setEnabled(hasImage && m_trainingResultIndex + 1 < m_trainingResultImages.size());
    m_trainingResultPageLabel->setText(hasImage
        ? QString("%1 / %2").arg(m_trainingResultIndex + 1).arg(m_trainingResultImages.size())
        : QStringLiteral("0 / 0"));
    if (!hasImage) {
        m_trainingResultNameLabel->setText(tr("暂无训练结果"));
        m_trainingResultImageLabel->setPixmap({});
        m_trainingResultImageLabel->setText(tr("训练完成后将在此展示结果图片"));
        return;
    }

    const QFileInfo imageInfo(m_trainingResultImages.at(m_trainingResultIndex));
    m_trainingResultNameLabel->setText(
        QString("%1 — %2").arg(imageInfo.dir().dirName(), imageInfo.fileName()));
    const QPixmap image(imageInfo.absoluteFilePath());
    if (image.isNull()) {
        m_trainingResultImageLabel->setPixmap({});
        m_trainingResultImageLabel->setText(tr("无法读取结果图片：%1").arg(imageInfo.fileName()));
        return;
    }
    const QSize target = m_trainingResultImageLabel->size().expandedTo(QSize(320, 240));
    m_trainingResultImageLabel->setText({});
    m_trainingResultImageLabel->setPixmap(
        image.scaled(target, Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::processFinished(int exitCode, QProcess::ExitStatus status)
{
    readProcessOutput();
    if (!m_trainBuffer.isEmpty()) {
        handleTrainingMessage(m_trainBuffer);
        m_trainBuffer.clear();
    }
    m_trainBtn->setEnabled(true);m_stopBtn->setEnabled(false);
    if((status!=QProcess::NormalExit || exitCode!=0) && m_trainingTaskId>=0){QSqlQuery q(m_trainingDb);q.prepare("UPDATE training_tasks SET status='failed',finished_at=? WHERE id=?");q.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));q.addBindValue(m_trainingTaskId);q.exec();m_logOutput->appendPlainText(tr("训练进程异常结束，退出码 %1").arg(exitCode));}
}

void MainWindow::initDatabase()
{
    const QString dataDir=QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);QDir().mkpath(dataDir);
    m_trainingDb=QSqlDatabase::addDatabase("QSQLITE","training_connection");m_trainingDb.setDatabaseName(QDir(dataDir).filePath("yolo_results.db"));
    if(!m_trainingDb.open()){statusBar()->showMessage(tr("训练数据库打开失败：%1").arg(m_trainingDb.lastError().text()));return;}
    QSqlQuery q(m_trainingDb);q.exec("CREATE TABLE IF NOT EXISTS training_tasks(id INTEGER PRIMARY KEY AUTOINCREMENT,started_at TEXT,finished_at TEXT,status TEXT,data_yaml TEXT,model TEXT,output_dir TEXT,best_model TEXT,last_model TEXT)");
    q.exec("CREATE TABLE IF NOT EXISTS training_metrics(id INTEGER PRIMARY KEY AUTOINCREMENT,task_id INTEGER,epoch INTEGER,name TEXT,value REAL,FOREIGN KEY(task_id) REFERENCES training_tasks(id))");
}
