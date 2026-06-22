#include "quantdialog.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QSettings>
#include <QSqlError>
#include <QSqlQuery>
#include <QSqlTableModel>
#include <QStandardPaths>
#include <QTableWidget>
#include <QTableView>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QPieSeries>

QuantDialog::QuantDialog(QWidget *parent) : QWidget(parent)
{
    initUI();
    initDatabase();
    loadSettings();
    m_process = new QProcess(this);
    m_process->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_process, &QProcess::readyReadStandardOutput, this, &QuantDialog::readProcessOutput);
    connect(m_process, QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished), this, &QuantDialog::processFinished);
}

QuantDialog::~QuantDialog()
{
    saveSettings();
    if (m_process && m_process->state() != QProcess::NotRunning) m_process->kill();
    if (m_db.isOpen()) m_db.close();
}

void QuantDialog::initUI()
{
    setWindowTitle(tr("模型量化与推理"));
    auto *main = new QVBoxLayout(this);
    auto addFileRow = [this](QVBoxLayout *layout, const QString &label, QLineEdit *&edit, const char *slot) {
        auto *row=new QHBoxLayout;edit=new QLineEdit;auto *button=new QPushButton(tr("浏览"));
        connect(button,SIGNAL(clicked()),this,slot);row->addWidget(new QLabel(label));row->addWidget(edit,1);row->addWidget(button);layout->addLayout(row);
    };

    auto *exportGroup=new QGroupBox(tr("模型导出 / 后训练量化"));auto *exportLayout=new QVBoxLayout(exportGroup);
    addFileRow(exportLayout,tr("源模型"),m_ptModelEdit,SLOT(selectPtModel()));
    auto *formatRow=new QHBoxLayout;m_formatCombo=new QComboBox;m_formatCombo->addItems({"onnx","engine","openvino","ncnn"});
    m_halfCheck=new QCheckBox("FP16");m_int8Check=new QCheckBox("INT8");formatRow->addWidget(new QLabel(tr("格式")));formatRow->addWidget(m_formatCombo);formatRow->addWidget(m_halfCheck);formatRow->addWidget(m_int8Check);formatRow->addStretch();exportLayout->addLayout(formatRow);
    addFileRow(exportLayout,tr("INT8 校准 data.yaml"),m_dataYamlEdit,SLOT(selectDataYaml()));
    m_quantBtn=new QPushButton(tr("开始导出/量化"));connect(m_quantBtn,&QPushButton::clicked,this,&QuantDialog::startQuantization);exportLayout->addWidget(m_quantBtn);main->addWidget(exportGroup);

    auto *inferGroup=new QGroupBox(tr("模型推理"));auto *inferLayout=new QVBoxLayout(inferGroup);
    addFileRow(inferLayout,tr("模型"),m_quantModelEdit,SLOT(selectQuantModel()));
    auto *sourceRow=new QHBoxLayout;m_inferSourceEdit=new QLineEdit;auto *fileBtn=new QPushButton(tr("选择图片"));auto *dirBtn=new QPushButton(tr("选择目录"));
    connect(fileBtn,&QPushButton::clicked,this,&QuantDialog::selectInferFile);connect(dirBtn,&QPushButton::clicked,this,&QuantDialog::selectInferDirectory);
    sourceRow->addWidget(new QLabel(tr("输入")));sourceRow->addWidget(m_inferSourceEdit,1);sourceRow->addWidget(fileBtn);sourceRow->addWidget(dirBtn);inferLayout->addLayout(sourceRow);
    auto *options=new QHBoxLayout;m_backendCombo=new QComboBox;m_backendCombo->addItem(tr("Python / Ultralytics"),"python");m_backendCombo->addItem(tr("C++ / ONNX Runtime"),"cpp");
    m_confSpin=new QDoubleSpinBox;m_confSpin->setRange(0.01,1);m_confSpin->setValue(0.25);m_confSpin->setSingleStep(0.05);
    m_iouSpin=new QDoubleSpinBox;m_iouSpin->setRange(0.01,1);m_iouSpin->setValue(0.45);m_iouSpin->setSingleStep(0.05);
    options->addWidget(new QLabel(tr("后端")));options->addWidget(m_backendCombo);options->addWidget(new QLabel("Confidence"));options->addWidget(m_confSpin);options->addWidget(new QLabel("IoU"));options->addWidget(m_iouSpin);options->addStretch();inferLayout->addLayout(options);
    auto *runtime=new QHBoxLayout;m_pythonEdit=new QLineEdit;m_ultralyticsEdit=new QLineEdit;m_outputEdit=new QLineEdit;auto *pythonBtn=new QPushButton(tr("Python…"));auto *outputBtn=new QPushButton(tr("输出…"));
    connect(pythonBtn,&QPushButton::clicked,this,&QuantDialog::selectPython);connect(outputBtn,&QPushButton::clicked,this,&QuantDialog::selectOutputDirectory);
    runtime->addWidget(new QLabel(tr("Python")));runtime->addWidget(m_pythonEdit);runtime->addWidget(pythonBtn);runtime->addWidget(new QLabel(tr("Ultralytics 源码")));runtime->addWidget(m_ultralyticsEdit);runtime->addWidget(new QLabel(tr("输出")));runtime->addWidget(m_outputEdit);runtime->addWidget(outputBtn);inferLayout->addLayout(runtime);
    auto *actions=new QHBoxLayout;m_inferBtn=new QPushButton(tr("开始推理"));m_dbBtn=new QPushButton(tr("检测明细"));m_statBtn=new QPushButton(tr("类别统计"));actions->addWidget(m_inferBtn);actions->addWidget(m_dbBtn);actions->addWidget(m_statBtn);inferLayout->addLayout(actions);main->addWidget(inferGroup);
    connect(m_inferBtn,&QPushButton::clicked,this,&QuantDialog::startInference);connect(m_dbBtn,&QPushButton::clicked,this,&QuantDialog::showDatabaseTable);connect(m_statBtn,&QPushButton::clicked,this,&QuantDialog::showStatistics);

    auto *bottom=new QHBoxLayout;
    auto *details=new QVBoxLayout;
    m_logOutput=new QTextEdit;m_logOutput->setReadOnly(true);details->addWidget(m_logOutput,1);
    details->addWidget(new QLabel(tr("当前图片检测框（点击记录高亮）")));
    m_detectionTable=new QTableWidget(0,6);
    m_detectionTable->setHorizontalHeaderLabels({tr("类别"),tr("置信度"),"x1","y1","x2","y2"});
    m_detectionTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_detectionTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_detectionTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_detectionTable->verticalHeader()->setVisible(false);
    m_detectionTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::Stretch);
    for(int column=1;column<6;++column)m_detectionTable->horizontalHeader()->setSectionResizeMode(column,QHeaderView::ResizeToContents);
    details->addWidget(m_detectionTable,1);bottom->addLayout(details,1);
    auto *viewer=new QVBoxLayout;m_resultImageLabel=new QLabel(tr("推理结果将在此显示"));m_resultImageLabel->setMinimumSize(480,320);m_resultImageLabel->setAlignment(Qt::AlignCenter);m_resultImageLabel->setStyleSheet("background:#202020;color:white");viewer->addWidget(m_resultImageLabel,1);
    auto *pages=new QHBoxLayout;m_prevBtn=new QPushButton(tr("上一张"));m_nextBtn=new QPushButton(tr("下一张"));m_pageLabel=new QLabel("0 / 0");pages->addWidget(m_prevBtn);pages->addWidget(m_pageLabel,1);pages->addWidget(m_nextBtn);viewer->addLayout(pages);bottom->addLayout(viewer,1);main->addLayout(bottom,1);
    connect(m_prevBtn,&QPushButton::clicked,this,&QuantDialog::showPreviousResult);connect(m_nextBtn,&QPushButton::clicked,this,&QuantDialog::showNextResult);
    connect(m_detectionTable,&QTableWidget::cellClicked,this,&QuantDialog::highlightDetection);
}

void QuantDialog::loadSettings(){QSettings s("MyAnnotator","YoloTools");m_pythonEdit->setText(s.value("python","python").toString());m_ultralyticsEdit->setText(s.value("ultralyticsRoot").toString());m_outputEdit->setText(s.value("outputDir",QDir::home().filePath("YOLORuns")).toString());}
void QuantDialog::saveSettings(){QSettings s("MyAnnotator","YoloTools");s.setValue("python",m_pythonEdit->text());s.setValue("ultralyticsRoot",m_ultralyticsEdit->text());s.setValue("outputDir",m_outputEdit->text());}
QString QuantDialog::bridgeScript()const{return QDir(QCoreApplication::applicationDirPath()).filePath("qt_quant_infer_bridge.py");}
void QuantDialog::selectPtModel(){m_ptModelEdit->setText(QFileDialog::getOpenFileName(this,tr("选择 PT 模型"),{},"PyTorch (*.pt)"));}
void QuantDialog::selectDataYaml(){m_dataYamlEdit->setText(QFileDialog::getOpenFileName(this,tr("选择 data.yaml"),{},"YAML (*.yaml *.yml)"));}
void QuantDialog::selectQuantModel(){m_quantModelEdit->setText(QFileDialog::getOpenFileName(this,tr("选择模型"),{},"Models (*.onnx *.engine *.xml *.pt);;All files (*)"));}
void QuantDialog::selectInferFile(){const QString v=QFileDialog::getOpenFileName(this,tr("选择图片"),{},"Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)");if(!v.isEmpty())m_inferSourceEdit->setText(v);}
void QuantDialog::selectInferDirectory(){const QString v=QFileDialog::getExistingDirectory(this,tr("选择图片目录"));if(!v.isEmpty())m_inferSourceEdit->setText(v);}
void QuantDialog::selectPython(){const QString v=QFileDialog::getOpenFileName(this,tr("选择 Python"),m_pythonEdit->text(),"Python (python.exe python3.exe);;All files (*)");if(!v.isEmpty())m_pythonEdit->setText(v);}
void QuantDialog::selectOutputDirectory(){const QString v=QFileDialog::getExistingDirectory(this,tr("选择输出目录"),m_outputEdit->text());if(!v.isEmpty())m_outputEdit->setText(v);}
void QuantDialog::setBusy(bool busy){m_quantBtn->setEnabled(!busy);m_inferBtn->setEnabled(!busy);}

void QuantDialog::startQuantization()
{
    if(m_ptModelEdit->text().isEmpty()){QMessageBox::warning(this,tr("缺少模型"),tr("请选择源模型。"));return;}
    if(m_halfCheck->isChecked()&&m_int8Check->isChecked()){QMessageBox::warning(this,tr("精度冲突"),tr("FP16 与 INT8 不能同时启用。"));return;}
    if(m_int8Check->isChecked()&&m_dataYamlEdit->text().isEmpty()){QMessageBox::warning(this,tr("缺少校准集"),tr("INT8 量化需要 data.yaml。"));return;}
    const QString script=bridgeScript();if(!QFileInfo::exists(script)){QMessageBox::warning(this,tr("缺少脚本"),script);return;}
    saveSettings();m_buffer.clear();m_processIsInference=false;setBusy(true);QStringList args{"-u",script,"--mode","export","--weights",m_ptModelEdit->text(),"--format",m_formatCombo->currentText(),"--project",m_outputEdit->text()};
    if(m_halfCheck->isChecked())args<<"--half";if(m_int8Check->isChecked())args<<"--int8"<<"--data"<<m_dataYamlEdit->text();if(!m_ultralyticsEdit->text().isEmpty())args<<"--ultralytics-root"<<m_ultralyticsEdit->text();m_process->start(m_pythonEdit->text().isEmpty()?"python":m_pythonEdit->text(),args);
}

void QuantDialog::startInference()
{
    if(m_quantModelEdit->text().isEmpty()||m_inferSourceEdit->text().isEmpty()){QMessageBox::warning(this,tr("缺少输入"),tr("请选择模型和图片/目录。"));return;}
    saveSettings();m_resultImages.clear();m_currentResultIndex=-1;m_pageLabel->setText("0 / 0");
    QSqlQuery q(m_db);q.prepare("INSERT INTO inference_tasks(started_at,status,backend,model,source) VALUES(?,?,?,?,?)");q.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));q.addBindValue("running");q.addBindValue(m_backendCombo->currentData());q.addBindValue(m_quantModelEdit->text());q.addBindValue(m_inferSourceEdit->text());q.exec();m_inferenceTaskId=q.lastInsertId().toLongLong();
    if(m_backendCombo->currentData().toString()=="cpp"){runCppInference();return;}
    const QString script=bridgeScript();if(!QFileInfo::exists(script)){QMessageBox::warning(this,tr("缺少脚本"),script);QSqlQuery failed(m_db);failed.exec(QString("UPDATE inference_tasks SET status='failed' WHERE id=%1").arg(m_inferenceTaskId));return;}m_buffer.clear();m_processIsInference=true;setBusy(true);
    QStringList args{"-u",script,"--mode","infer","--weights",m_quantModelEdit->text(),"--source",m_inferSourceEdit->text(),"--project",m_outputEdit->text(),"--conf",QString::number(m_confSpin->value()),"--iou",QString::number(m_iouSpin->value())};
    if(!m_ultralyticsEdit->text().isEmpty())args<<"--ultralytics-root"<<m_ultralyticsEdit->text();m_process->start(m_pythonEdit->text().isEmpty()?"python":m_pythonEdit->text(),args);
}

void QuantDialog::readProcessOutput(){m_buffer+=m_process->readAllStandardOutput();int p;while((p=m_buffer.indexOf('\n'))>=0){handleMessage(m_buffer.left(p).trimmed());m_buffer.remove(0,p+1);}}
void QuantDialog::handleMessage(const QByteArray &line)
{
    if(line.isEmpty())return;QJsonParseError e;const auto doc=QJsonDocument::fromJson(line,&e);if(e.error!=QJsonParseError::NoError||!doc.isObject()){m_logOutput->append(QString::fromUtf8(line));return;}const auto o=doc.object();const QString event=o.value("event").toString();
    if(event=="detection"){const auto xy=o.value("xyxy").toArray();QSqlQuery q(m_db);q.prepare("INSERT INTO detections(task_id,image_name,class_id,class_name,confidence,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?,?,?)");q.addBindValue(m_inferenceTaskId);q.addBindValue(o.value("image").toString());q.addBindValue(o.value("class_id").toInt());q.addBindValue(o.value("class_name").toString());q.addBindValue(o.value("confidence").toDouble());for(int i=0;i<4;++i)q.addBindValue(i<xy.size()?xy.at(i).toDouble():0.0);q.exec();}
    else if(event=="infer_completed"){m_currentResultDir=o.value("result_dir").toString();loadResultDirectory(m_currentResultDir);QSqlQuery q(m_db);q.prepare("UPDATE inference_tasks SET status='completed',finished_at=?,result_dir=? WHERE id=?");q.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));q.addBindValue(m_currentResultDir);q.addBindValue(m_inferenceTaskId);q.exec();}
    else if(event=="export_completed")m_logOutput->append(tr("模型已导出：%1").arg(o.value("path").toString()));else if(event=="error")m_logOutput->append(tr("错误：%1").arg(o.value("message").toString()));else m_logOutput->append(QString::fromUtf8(line));
}

void QuantDialog::processFinished(int exitCode,QProcess::ExitStatus status){setBusy(false);if(status!=QProcess::NormalExit||exitCode!=0){if(m_processIsInference){QSqlQuery q(m_db);q.prepare("UPDATE inference_tasks SET status='failed',finished_at=? WHERE id=?");q.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));q.addBindValue(m_inferenceTaskId);q.exec();}m_logOutput->append(tr("进程异常结束：%1").arg(exitCode));}}

void QuantDialog::runCppInference()
{
    if(!m_onnx.isAvailable()){QMessageBox::warning(this,tr("ONNX Runtime 未启用"),tr("请设置 ONNXRUNTIME_ROOT 后重新配置并构建项目。"));QSqlQuery failed(m_db);failed.exec(QString("UPDATE inference_tasks SET status='failed' WHERE id=%1").arg(m_inferenceTaskId));return;}
    QString error;if(!m_onnx.load(m_quantModelEdit->text(),&error)){QMessageBox::warning(this,tr("模型加载失败"),error);QSqlQuery failed(m_db);failed.exec(QString("UPDATE inference_tasks SET status='failed' WHERE id=%1").arg(m_inferenceTaskId));return;}
    QSettings settings("MyAnnotator","YoloTools");m_onnx.setClassNames(settings.value("classes").toStringList());
    QStringList files;QFileInfo input(m_inferSourceEdit->text());if(input.isFile())files<<input.absoluteFilePath();else{QDir dir(input.absoluteFilePath());const QStringList names=dir.entryList({"*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"},QDir::Files,QDir::Name);for(const QString &name:names)files<<dir.filePath(name);}
    if(files.isEmpty()){QMessageBox::warning(this,tr("没有图片"),tr("输入中没有支持的图片。"));return;}setBusy(true);QApplication::setOverrideCursor(Qt::WaitCursor);
    m_currentResultDir=QDir(m_outputEdit->text()).filePath(QString("infer_cpp_%1").arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss")));QDir().mkpath(m_currentResultDir);
    for(const QString &file:files){QImage image(file);const auto detections=m_onnx.infer(image,float(m_confSpin->value()),float(m_iouSpin->value()),&error);if(!error.isEmpty()){m_logOutput->append(error);error.clear();continue;}QPainter painter(&image);painter.setPen(QPen(Qt::red,2));
        for(const auto &d:detections){painter.drawRect(d.rect);painter.drawText(d.rect.topLeft()+QPointF(0,-3),QString("%1 %2").arg(d.className).arg(d.confidence,0,'f',2));QSqlQuery q(m_db);q.prepare("INSERT INTO detections(task_id,image_name,class_id,class_name,confidence,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?,?,?)");q.addBindValue(m_inferenceTaskId);q.addBindValue(QFileInfo(file).fileName());q.addBindValue(d.classId);q.addBindValue(d.className);q.addBindValue(d.confidence);q.addBindValue(d.rect.left());q.addBindValue(d.rect.top());q.addBindValue(d.rect.right());q.addBindValue(d.rect.bottom());q.exec();}painter.end();image.save(QDir(m_currentResultDir).filePath(QFileInfo(file).fileName()));}
    QApplication::restoreOverrideCursor();setBusy(false);loadResultDirectory(m_currentResultDir);QSqlQuery q(m_db);q.prepare("UPDATE inference_tasks SET status='completed',finished_at=?,result_dir=? WHERE id=?");q.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODate));q.addBindValue(m_currentResultDir);q.addBindValue(m_inferenceTaskId);q.exec();
}

void QuantDialog::loadResultDirectory(const QString &directory){QDir dir(directory);m_resultImages.clear();for(const QString &name:dir.entryList({"*.jpg","*.jpeg","*.png","*.bmp","*.webp"},QDir::Files,QDir::Name))m_resultImages<<dir.filePath(name);m_currentResultIndex=m_resultImages.isEmpty()?-1:0;updateResultImage();}
void QuantDialog::showPreviousResult(){if(m_currentResultIndex>0){--m_currentResultIndex;updateResultImage();}}
void QuantDialog::showNextResult(){if(m_currentResultIndex+1<m_resultImages.size()){++m_currentResultIndex;updateResultImage();}}

void QuantDialog::updateResultImage()
{
    m_selectedDetectionId=-1;
    refreshDetectionTable();
    renderResultImage();
    if(m_currentResultIndex<0){m_pageLabel->setText("0 / 0");return;}
    m_pageLabel->setText(QString("%1 / %2").arg(m_currentResultIndex+1).arg(m_resultImages.size()));
    m_prevBtn->setEnabled(m_currentResultIndex>0);
    m_nextBtn->setEnabled(m_currentResultIndex+1<m_resultImages.size());
}

void QuantDialog::refreshDetectionTable()
{
    m_detectionTable->blockSignals(true);
    m_detectionTable->setRowCount(0);
    if(m_currentResultIndex<0||m_currentResultIndex>=m_resultImages.size()||m_inferenceTaskId<0){m_detectionTable->blockSignals(false);return;}
    QSqlQuery query(m_db);
    query.prepare("SELECT id,class_name,confidence,x1,y1,x2,y2 FROM detections WHERE task_id=? AND image_name=? ORDER BY confidence DESC,id");
    query.addBindValue(m_inferenceTaskId);
    query.addBindValue(QFileInfo(m_resultImages[m_currentResultIndex]).fileName());
    if(query.exec())while(query.next()){
        const int row=m_detectionTable->rowCount();m_detectionTable->insertRow(row);
        auto *classItem=new QTableWidgetItem(query.value(1).toString());classItem->setData(Qt::UserRole,query.value(0));m_detectionTable->setItem(row,0,classItem);
        m_detectionTable->setItem(row,1,new QTableWidgetItem(QString::number(query.value(2).toDouble(),'f',3)));
        for(int column=2;column<6;++column)m_detectionTable->setItem(row,column,new QTableWidgetItem(QString::number(query.value(column+1).toDouble(),'f',1)));
    }
    m_detectionTable->blockSignals(false);
}

void QuantDialog::renderResultImage()
{
    if(m_currentResultIndex<0||m_currentResultIndex>=m_resultImages.size()){
        m_resultImageLabel->clear();m_resultImageLabel->setText(tr("没有结果图片"));return;
    }
    QPixmap pixmap(m_resultImages[m_currentResultIndex]);
    if(pixmap.isNull()){m_resultImageLabel->clear();m_resultImageLabel->setText(tr("结果图片加载失败"));return;}
    if(m_selectedDetectionId>=0){
        QSqlQuery query(m_db);query.prepare("SELECT class_name,confidence,x1,y1,x2,y2 FROM detections WHERE id=? AND task_id=? AND image_name=?");
        query.addBindValue(m_selectedDetectionId);query.addBindValue(m_inferenceTaskId);query.addBindValue(QFileInfo(m_resultImages[m_currentResultIndex]).fileName());
        if(query.exec()&&query.next()){
            const QRectF rect(QPointF(query.value(2).toDouble(),query.value(3).toDouble()),QPointF(query.value(4).toDouble(),query.value(5).toDouble()));
            const QString text=QString("%1 %2").arg(query.value(0).toString()).arg(query.value(1).toDouble(),0,'f',2);
            QPainter painter(&pixmap);painter.setRenderHint(QPainter::Antialiasing);
            const QColor highlight(0,255,255);painter.setPen(QPen(highlight,5));painter.setBrush(QColor(0,255,255,35));painter.drawRect(rect);
            QFont font=painter.font();font.setBold(true);font.setPointSize(qMax(10,font.pointSize()));painter.setFont(font);
            const QRect textRect=painter.fontMetrics().boundingRect(text).adjusted(-5,-3,5,3);
            QRect labelRect=textRect;labelRect.moveBottomLeft(rect.topLeft().toPoint()+QPoint(0,-2));
            if(labelRect.top()<0)labelRect.moveTopLeft(rect.topLeft().toPoint()+QPoint(0,2));
            painter.fillRect(labelRect,highlight);painter.setPen(Qt::black);painter.drawText(labelRect,Qt::AlignCenter,text);
        }
    }
    m_resultImageLabel->setPixmap(pixmap.scaled(m_resultImageLabel->size(),Qt::KeepAspectRatio,Qt::SmoothTransformation));
}

void QuantDialog::highlightDetection(int row,int column)
{
    Q_UNUSED(column)
    const QTableWidgetItem *item=m_detectionTable->item(row,0);
    if(!item)return;
    m_selectedDetectionId=item->data(Qt::UserRole).toLongLong();
    renderResultImage();
}

void QuantDialog::initDatabase(){const QString dir=QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);QDir().mkpath(dir);m_db=QSqlDatabase::addDatabase("QSQLITE","inference_connection");m_db.setDatabaseName(QDir(dir).filePath("yolo_results.db"));if(!m_db.open()){m_logOutput->append(m_db.lastError().text());return;}QSqlQuery q(m_db);q.exec("CREATE TABLE IF NOT EXISTS inference_tasks(id INTEGER PRIMARY KEY AUTOINCREMENT,started_at TEXT,finished_at TEXT,status TEXT,backend TEXT,model TEXT,source TEXT,result_dir TEXT)");q.exec("CREATE TABLE IF NOT EXISTS detections(id INTEGER PRIMARY KEY AUTOINCREMENT,task_id INTEGER,image_name TEXT,class_id INTEGER,class_name TEXT,confidence REAL,x1 REAL,y1 REAL,x2 REAL,y2 REAL,timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)");}
void QuantDialog::showStatistics(){QSqlQuery q(m_db);q.prepare("SELECT class_name,COUNT(*) FROM detections WHERE task_id=? GROUP BY class_name");q.addBindValue(m_inferenceTaskId);q.exec();auto *series=new QPieSeries;while(q.next())series->append(QString("%1 (%2)").arg(q.value(0).toString()).arg(q.value(1).toInt()),q.value(1).toInt());if(series->count()==0){delete series;QMessageBox::information(this,tr("无数据"),tr("当前推理任务没有检测记录。"));return;}auto *chart=new QChart;chart->addSeries(series);chart->setTitle(tr("当前推理任务类别统计"));auto *dialog=new QDialog(this);auto *layout=new QVBoxLayout(dialog);layout->addWidget(new QChartView(chart));dialog->resize(720,520);dialog->exec();}
void QuantDialog::showDatabaseTable(){auto *dialog=new QDialog(this);auto *layout=new QVBoxLayout(dialog);auto *view=new QTableView;auto *model=new QSqlTableModel(dialog,m_db);model->setTable("detections");model->setFilter(QString("task_id=%1").arg(m_inferenceTaskId));model->select();view->setModel(model);view->setEditTriggers(QAbstractItemView::NoEditTriggers);view->horizontalHeader()->setStretchLastSection(true);layout->addWidget(view);dialog->resize(950,520);dialog->exec();}
