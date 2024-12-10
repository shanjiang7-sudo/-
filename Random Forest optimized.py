import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split

# 设置图表显示模式为 TKAgg
plt.switch_backend('TkAgg')


# 读取 .dat 文件的函数
def load_bat_files_from_folder(folder_path, expected_columns=52, max_len=100):
    data = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".dat"):
            file_path = os.path.join(folder_path, filename)
            file_data = []
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f.readlines()):
                    try:
                        values = list(map(float, line.strip().split()))
                        if len(values) < expected_columns:
                            values.extend([0] * (expected_columns - len(values)))
                        elif len(values) > expected_columns:
                            values = values[:expected_columns]
                        file_data.append(values)
                    except ValueError:
                        pass
            if len(file_data) < max_len:
                file_data = np.pad(file_data, ((0, max_len - len(file_data)), (0, 0)), mode='constant',
                                   constant_values=0)
            elif len(file_data) > max_len:
                file_data = file_data[:max_len]
            data.append(np.array(file_data))
    return data


# 使用路径
train_folder = r'C:\Users\86150\PycharmProjects\pythonProject2\TE化工数据集\训练集'
test_folder = r'C:\Users\86150\PycharmProjects\pythonProject2\TE化工数据集\测试集'

# 加载数据
train_data = load_bat_files_from_folder(train_folder, expected_columns=52, max_len=500)
test_data = load_bat_files_from_folder(test_folder, expected_columns=52, max_len=1000)


# 生成标签
def generate_labels(data, start_label=0):
    labels = []
    for file_idx, file_data in enumerate(data):
        num_samples = file_data.shape[0]
        labels.extend([start_label + file_idx] * num_samples)
    return np.array(labels)


# 生成训练集和测试集标签
X_train = np.vstack(train_data)
y_train = generate_labels(train_data, start_label=0)
X_test = np.vstack(test_data)
y_test = generate_labels(test_data, start_label=0)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 随机森林超参数优化
def optimize_random_forest(X_train, y_train):
    # 设置超参数范围
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # 使用GridSearchCV进行超参数调优
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print(f"最佳超参数: {grid_search.best_params_}")
    return grid_search.best_estimator_


# 优化后的随机森林模型
rf_model_optimized = optimize_random_forest(X_train, y_train)


# 训练过程可视化函数
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name="Model"):
    train_accuracies = []
    val_accuracies = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_fold, y_train_fold)

        # 计算训练集和验证集准确率
        train_accuracy = model.score(X_train_fold, y_train_fold)
        val_accuracy = model.score(X_val_fold, y_val_fold)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # 可视化准确率
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Train Accuracy', color='blue', marker='o')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange', marker='x')
    plt.title(f'{model_name} - Train & Validation Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# 模型初始化
svm_model = SVC(kernel='linear', random_state=42)
knn_model = KNeighborsClassifier()
kmeans = KMeans(n_clusters=len(np.unique(y_train)), random_state=42)

# 训练并评估模型
train_and_evaluate_model(svm_model, X_train, y_train, X_test, y_test, model_name="SVM")
train_and_evaluate_model(rf_model_optimized, X_train, y_train, X_test, y_test, model_name="Optimized Random Forest")
train_and_evaluate_model(knn_model, X_train, y_train, X_test, y_test, model_name="KNN")

# KMeans无监督学习
kmeans.fit(X_train)
y_pred_kmeans = kmeans.predict(X_test)
print(f"KMeans 聚类报告:\n{classification_report(y_test, y_pred_kmeans)}")


# 混淆矩阵可视化函数
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"混淆矩阵 - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))

    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('实际标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.show()


# 评估每个模型
# SVM模型评估
y_pred_svm = svm_model.predict(X_test)
print(f"SVM 分类报告:\n{classification_report(y_test, y_pred_svm)}")
plot_confusion_matrix(y_test, y_pred_svm, "SVM")

# 随机森林评估
y_pred_rf = rf_model_optimized.predict(X_test)
print(f"Optimized Random Forest 分类报告:\n{classification_report(y_test, y_pred_rf)}")
plot_confusion_matrix(y_test, y_pred_rf, "Optimized Random Forest")

# KNN评估
y_pred_knn = knn_model.predict(X_test)
print(f"KNN 分类报告:\n{classification_report(y_test, y_pred_knn)}")
plot_confusion_matrix(y_test, y_pred_knn, "KNN")

# KMeans评估
print(f"KMeans 分类报告:\n{classification_report(y_test, y_pred_kmeans)}")
plot_confusion_matrix(y_test, y_pred_kmeans, "KMeans")
