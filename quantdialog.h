#ifndef QUANTDIALOG_H
#define QUANTDIALOG_H

#include "onnxinferenceengine.h"

#include <QWidget>
#include <QProcess>
#include <QSqlDatabase>
#include <QStringList>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTableWidget;
class QTextEdit;

class QuantDialog : public QWidget
{
    Q_OBJECT
public:
    explicit QuantDialog(QWidget *parent = nullptr);
    ~QuantDialog() override;

private slots:
    void selectPtModel();
    void selectDataYaml();
    void selectQuantModel();
    void selectInferFile();
    void selectInferDirectory();
    void selectPython();
    void selectOutputDirectory();
    void startQuantization();
    void startInference();
    void readProcessOutput();
    void processFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void showPreviousResult();
    void showNextResult();
    void showStatistics();
    void showDatabaseTable();
    void highlightDetection(int row, int column);

private:
    void initUI();
    void initDatabase();
    void loadSettings();
    void saveSettings();
    void handleMessage(const QByteArray &line);
    void loadResultDirectory(const QString &directory);
    void updateResultImage();
    void refreshDetectionTable();
    void renderResultImage();
    void runCppInference();
    QString bridgeScript() const;
    void setBusy(bool busy);

    QLineEdit *m_ptModelEdit = nullptr;
    QLineEdit *m_dataYamlEdit = nullptr;
    QComboBox *m_formatCombo = nullptr;
    QCheckBox *m_halfCheck = nullptr;
    QCheckBox *m_int8Check = nullptr;
    QPushButton *m_quantBtn = nullptr;
    QLineEdit *m_quantModelEdit = nullptr;
    QLineEdit *m_inferSourceEdit = nullptr;
    QLineEdit *m_pythonEdit = nullptr;
    QLineEdit *m_ultralyticsEdit = nullptr;
    QLineEdit *m_outputEdit = nullptr;
    QComboBox *m_backendCombo = nullptr;
    QDoubleSpinBox *m_confSpin = nullptr;
    QDoubleSpinBox *m_iouSpin = nullptr;
    QPushButton *m_inferBtn = nullptr;
    QPushButton *m_statBtn = nullptr;
    QPushButton *m_dbBtn = nullptr;
    QTextEdit *m_logOutput = nullptr;
    QTableWidget *m_detectionTable = nullptr;
    QLabel *m_resultImageLabel = nullptr;
    QPushButton *m_prevBtn = nullptr;
    QPushButton *m_nextBtn = nullptr;
    QLabel *m_pageLabel = nullptr;
    QProcess *m_process = nullptr;
    QByteArray m_buffer;
    QString m_currentResultDir;
    QStringList m_resultImages;
    int m_currentResultIndex = -1;
    qint64 m_selectedDetectionId = -1;
    qint64 m_inferenceTaskId = -1;
    bool m_processIsInference = false;
    QSqlDatabase m_db;
    OnnxInferenceEngine m_onnx;
};

#endif
