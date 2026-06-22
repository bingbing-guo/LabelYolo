#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "canvaswidget.h"

#include <QMainWindow>
#include <QProcess>
#include <QSqlDatabase>
#include <QStringList>

class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QPlainTextEdit;
class QStackedWidget;
class QWidget;
class QAction;
class QuantDialog;

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void openFolder();
    void chooseSaveDir();
    void saveCurrentAnnotation();
    void addNewClass();
    void deleteCurrentClass();
    void redoCurrentAnnotation();
    void prevImage();
    void nextImage();
    void onImageSelected(int index);
    void onClassSelected(int comboIndex);
    void refreshAnnotationList();
    void chooseTrainDir();
    void chooseValDir();
    void choosePython();
    void chooseUltralyticsRoot();
    void chooseOutputDir();
    void chooseTrainingResultDir();
    void startTraining();
    void stopTraining();
    void readProcessOutput();
    void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void showAnnotationPage();
    void showTrainingPage();
    void showQuantizationPage();

private:
    void initUI();
    void initTrainingUI();
    void initDatabase();
    void loadSettings();
    void saveSettings();
    void rebuildClassCombo(int preferredId = -1);
    void updateDatasetSummary();
    bool createDatasetYaml(QString *path, QString *error) const;
    void handleTrainingMessage(const QByteArray &line);
    void loadTrainingResultImages(const QString &directory);
    void loadLatestTrainingResultImages();
    void loadTrainingSummary(const QString &directory);
    void updateTrainingResultImage();
    QString scriptPath(const QString &name) const;

    CanvasWidget *m_canvas = nullptr;
    QStackedWidget *m_pages = nullptr;
    QWidget *m_annotationPage = nullptr;
    QWidget *m_trainingPage = nullptr;
    QuantDialog *m_quantizationPage = nullptr;
    QAction *m_annotationPageAction = nullptr;
    QAction *m_trainingPageAction = nullptr;
    QAction *m_quantizationPageAction = nullptr;
    QListWidget *m_imageList = nullptr;
    QListWidget *m_annotationList = nullptr;
    QComboBox *m_classCombo = nullptr;
    QLabel *m_imageStatus = nullptr;
    QLabel *m_zoomStatus = nullptr;
    QLabel *m_annotationStatus = nullptr;
    QLabel *m_datasetSummary = nullptr;
    QStringList m_imagePaths;
    QStringList m_classList;
    int m_currentImageIndex = -1;
    QString m_saveDir;

    QProcess *m_trainProcess = nullptr;
    QLineEdit *m_trainDirEdit = nullptr;
    QLineEdit *m_valDirEdit = nullptr;
    QLineEdit *m_pythonEdit = nullptr;
    QLineEdit *m_ultralyticsEdit = nullptr;
    QLineEdit *m_outputEdit = nullptr;
    QLineEdit *m_modelEdit = nullptr;
    QLineEdit *m_deviceEdit = nullptr;
    QSpinBox *m_epochsSpin = nullptr;
    QSpinBox *m_batchSpin = nullptr;
    QSpinBox *m_imgSizeSpin = nullptr;
    QDoubleSpinBox *m_lrSpin = nullptr;
    QComboBox *m_optimizerCombo = nullptr;
    QPlainTextEdit *m_logOutput = nullptr;
    QPushButton *m_trainBtn = nullptr;
    QPushButton *m_stopBtn = nullptr;
    QProgressBar *m_trainProgress = nullptr;
    QTableWidget *m_trainingSummaryTable = nullptr;
    QLabel *m_trainingResultImageLabel = nullptr;
    QLabel *m_trainingResultNameLabel = nullptr;
    QLabel *m_trainingResultPageLabel = nullptr;
    QPushButton *m_trainingResultPrevBtn = nullptr;
    QPushButton *m_trainingResultNextBtn = nullptr;
    QStringList m_trainingResultImages;
    QString m_trainingResultDirectory;
    int m_trainingResultIndex = -1;
    QByteArray m_trainBuffer;
    qint64 m_trainingTaskId = -1;
    QSqlDatabase m_trainingDb;
};

#endif
