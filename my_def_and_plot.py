import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score, roc_curve
import plots as p
import phik
from sklearn.base import clone

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from sklearn.base import clone
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def train_evaluate_models_cv_SMOTE(models: list, X, y, preprocessor=None, cv=5, seed=None, smote_params=None):
    """
    models: список кортежей (name, model)
    preprocessor: ColumnTransformer / OneHotEncoder и т.п. (может быть None)
    smote_params: dict с параметрами SMOTE/SMOTENC, например:
        {'type': 'SMOTENC', 'categorical_features': [0,1,2], 'sampling_strategy': 0.4, 'k_neighbors': 5}
    """
    from imblearn.base import BaseSampler

    all_metrics = {}

    for model_name, base_model in models:
        current_model = clone(base_model)

        # 1. Если нужен SMOTE — оборачиваем его в пайплайн
        if smote_params is not None:
            smote_type = smote_params.get('type', 'SMOTENC')
            if smote_type == 'SMOTENC':
                from imblearn.over_sampling import SMOTENC
                smote = SMOTENC(
                    categorical_features=smote_params['categorical_features'],
                    sampling_strategy=smote_params.get('sampling_strategy', 0.5),
                    k_neighbors=smote_params.get('k_neighbors', 5),
                    random_state=seed
                )
            else:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(
                    sampling_strategy=smote_params.get('sampling_strategy', 0.5),
                    k_neighbors=smote_params.get('k_neighbors', 5),
                    random_state=seed
                )

            # Пайплайн: SMOTE -> (опционально preprocessor) -> модель
            steps = [('smote', smote)]
            if preprocessor is not None:
                steps.append(('preprocess', clone(preprocessor)))
            steps.append(('model', current_model))

            final_pipeline = ImbPipeline(steps=steps)
        else:
            # Без SMOTE: просто preprocessor + модель
            if preprocessor is not None:
                final_pipeline = ImbPipeline(steps=[
                    ('preprocess', clone(preprocessor)),
                    ('model', current_model)
                ])
            else:
                final_pipeline = current_model

        all_metrics[model_name] = train_evaluate_model_cv(
            final_pipeline, model_name, X, y, None, cv, seed
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".4f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df

def train_evaluate_model_cv(model, model_name, X, y,
                            preprocessor=None, cv=5, seed=None):
    """
    Train and evaluate a model using cross-validation and optional preprocessing.
    No Pipeline used — preprocessor is applied manually per fold.

    Args:
        model: The model to train and evaluate
        model_name: Name of the model for reporting
        X: Features (DataFrame or array)
        y: Target labels
        preprocessor: Preprocessing object with fit/transform, or None
        cv: Number of cross-validation folds or a CV splitter object
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing evaluation metrics
    """
    # Set random seed if model supports it
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Prepare CV splitter
    if isinstance(cv, int):
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        cv_splitter = cv

    # Check if X is DataFrame (use .iloc) or array (use direct indexing)
    is_dataframe = hasattr(X, 'iloc')
    is_series = hasattr(y, 'iloc')

    roc_auc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        y_train = y.iloc[train_idx] if is_series else y[train_idx]
        y_val = y.iloc[val_idx] if is_series else y[val_idx]

        # Apply preprocessor manually (fit on train, transform on val)
        if preprocessor is not None:
            X_train_proc = preprocessor.fit_transform(X_train)
            X_val_proc = preprocessor.transform(X_val)
        else:
            X_train_proc, X_val_proc = X_train, X_val

        # Clone model for this fold
        model_fold = clone(model)
        model_fold.fit(X_train_proc, y_train)

        y_pred = model_fold.predict(X_val_proc)

        # predict_proba есть не у всех моделей
        if hasattr(model_fold, 'predict_proba'):
            y_proba = model_fold.predict_proba(X_val_proc)[:, 1]
        else:
            y_proba = None

        accuracy_scores.append(accuracy_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, average='macro', zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, average='macro', zero_division=0))

        if y_proba is not None and len(np.unique(y_val)) > 1:
            roc_auc_scores.append(roc_auc_score(y_val, y_proba))
        else:
            roc_auc_scores.append(np.nan)

    metrics = {
        'ROC AUC': np.nanmean(roc_auc_scores),
        'F1 Score': np.mean(f1_scores),
        'Precision': np.mean(precision_scores),
        'Recall': np.mean(recall_scores),
        'Accuracy': np.mean(accuracy_scores),
    }

    # График (если p доступен)
    try:
        p.plot_classification_results(metrics, model_name)
    except NameError:
        pass

    return metrics

def train_evaluate_models_cv(models: list, X, y, preprocessor=None, cv=5, seed=None):
    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели
        current_model = clone(model)

        # Клонируем preprocessor только если он передан
        current_preprocessor = clone(preprocessor) if preprocessor is not None else None

        # Store metrics
        all_metrics[model_name] = train_evaluate_model_cv(
            current_model, model_name, X, y, current_preprocessor, cv, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df

def calculate_classification_metrics(y_test, y_pred, y_probs=None):
    """
    Calculate classification performance metrics (works with 0/1 or string labels)
    """
    # Определяем, какие метки есть и какой считать положительным
    unique_labels = np.unique(y_test)
    
    # Логика: если есть 0 и 1 — считаем 1 положительным; иначе считаем 'yes' положительным
    if set(unique_labels) == {0, 1}:
        pos_label = 1
        neg_label = 0
    elif 'yes' in unique_labels and 'no' in unique_labels:
        pos_label = 'yes'
        neg_label = 'no'
    else:
        # fallback: первый элемент как положительный, второй как отрицательный
        pos_label = unique_labels[0]
        neg_label = unique_labels[-1] if len(unique_labels) > 1 else None

    metrics = {
        'ROC AUC': roc_auc_score(y_test, y_probs) if y_probs is not None else None,
        'F1 Score': f1_score(y_test, y_pred, average='macro'),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'Accuracy': (y_pred == y_test).mean(),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

    # Метрики по классам с корректным pos_label
    metrics['Classification Report'] = {
        'Class': ['Positive', 'Negative'],
        'Precision': [
            precision_score(y_test, y_pred, pos_label=pos_label),
            precision_score(y_test, y_pred, pos_label=neg_label) if neg_label is not None else None
        ],
        'Recall': [
            recall_score(y_test, y_pred, pos_label=pos_label),
            recall_score(y_test, y_pred, pos_label=neg_label) if neg_label is not None else None
        ]
    }

    if y_probs is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_probs, pos_label=pos_label)
        metrics['ROC Curve'] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }

    return metrics


# функция для модели
def evaluate_classification(y_test, y_pred, y_probs=None, model_name="Model", enable_plot=True):
    """
    Evaluate classification performance with comprehensive metrics and visualizations

    Parameters:
    -----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_probs : array-like, optional
        Predicted probabilities for positive class (required for ROC AUC)
    model_name : str, optional
        Name of the model for display purposes
    enable_plot : bool, optional
        Whether to display plots and detailed reports

    Returns:
    --------
    dict: Dictionary containing all calculated metrics
    """
    # Calculate all metrics
    metrics = calculate_classification_metrics(y_test, y_pred, y_probs)

    if enable_plot:
        # Generate plots
        p.plot_classification_results(metrics, model_name)

        # Print detailed report
        p.print_classification_report(metrics, model_name)

    # Return metrics dictionary (excluding plot data for cleaner output)
    return {k: v for k, v in metrics.items() if k not in ['Confusion Matrix', 'ROC Curve', 'Classification Report']}

# функция разделения на входные признаки и таргет
def divide_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def plot_categorical_relationship(df, col1, col2, figsize=(18, 6)):
    """
    Строит 3 тепловые карты:
    1) абсолютные значения,
    2) доли внутри col1 (по строкам),
    3) доли внутри col2 (по столбцам).

    Автоматически подстраивает размер шрифта и поворачивает подписи,
    чтобы графики оставались читаемыми даже при многих категориях.
    """
    # 1. Считаем таблицы
    count = pd.crosstab(df[col1], df[col2])
    row_prop = pd.crosstab(df[col1], df[col2], normalize='index')
    col_prop = pd.crosstab(df[col1], df[col2], normalize='columns')

    # 2. Число категорий для настройки шрифта
    n_rows = len(count)
    n_cols = len(count.columns)

    # Подбираем размер шрифта в зависимости от числа категорий
    if n_rows > 15 or n_cols > 15:
        annot_fontsize = 6
        label_fontsize = 8
        tick_labelsize = 6
        annot = True
    elif n_rows > 8 or n_cols > 8:
        annot_fontsize = 8
        label_fontsize = 10
        tick_labelsize = 8
        annot = True
    else:
        annot_fontsize = 10
        label_fontsize = 12
        tick_labelsize = 10
        annot = True

    # Если очень много ячеек — отключаем аннотации, иначе будет «каша»
    if n_rows * n_cols > 200:
        annot = False

    # 3. Создаём фигуру
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 4. Абсолютные значения
    sns.heatmap(
        count,
        annot=annot,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        annot_kws={"fontsize": annot_fontsize},
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title(f'Абсолютные значения\n{col1} vs {col2}', fontsize=label_fontsize)
    axes[0].set_xlabel(col2, fontsize=label_fontsize)
    axes[0].set_ylabel(col1, fontsize=label_fontsize)
    axes[0].tick_params(axis='both', labelsize=tick_labelsize)
    axes[0].tick_params(axis='x', rotation=45)

    # 5. Доли внутри col1 (по строкам)
    sns.heatmap(
        row_prop,
        annot=annot,
        fmt=".2f",
        cmap="Greens",
        ax=axes[1],
        annot_kws={"fontsize": annot_fontsize},
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title(f'Доли внутри {col1}\n(по строкам)', fontsize=label_fontsize)
    axes[1].set_xlabel(col2, fontsize=label_fontsize)
    axes[1].set_ylabel(col1, fontsize=label_fontsize)
    axes[1].tick_params(axis='both', labelsize=tick_labelsize)
    axes[1].tick_params(axis='x', rotation=45)

    # 6. Доли внутри col2 (по столбцам)
    sns.heatmap(
        col_prop,
        annot=annot,
        fmt=".2f",
        cmap="Oranges",
        ax=axes[2],
        annot_kws={"fontsize": annot_fontsize},
        cbar_kws={"shrink": 0.8},
    )
    axes[2].set_title(f'Доли внутри {col2}\n(по столбцам)', fontsize=label_fontsize)
    axes[2].set_xlabel(col2, fontsize=label_fontsize)
    axes[2].set_ylabel(col1, fontsize=label_fontsize)
    axes[2].tick_params(axis='both', labelsize=tick_labelsize)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    
# импорт матрицы phik
def plot_phik(data, figsize=(12, 8)):
    phik_matrix = data.phik_matrix()
    plt.figure(figsize=(10, 10))
    sns.heatmap(phik_matrix, annot=True, fmt=".1f", cmap='coolwarm', cbar=True)
    plt.show()

# функция для вывода гистограммы (количесвтенный признак)
def plot_hist_numeric(data, feature, figsize=(10, 6), x_min=None, x_max=None, 
                      bins=30, color='#3498db', alpha=0.7, show_boxplot=True):
    """
    Построение гистограммы с KDE и box plot для числового признака.
    
    Параметры:
    ----------
    data : pd.DataFrame
        Исходный датафрейм
    feature : str
        Название столбца для визуализации
    figsize : tuple
        Размер графика (ширина, высота)
    x_min, x_max : float, optional
        Границы отсечения выбросов
    bins : int
        Количество бинов гистограммы
    color : str
        Цвет гистограммы
    alpha : float
        Прозрачность гистограммы
    show_boxplot : bool
        Показывать ли box plot
    """
    filtered_data = data.copy()
    if x_min is not None:
        filtered_data = filtered_data[filtered_data[feature] >= x_min]
    if x_max is not None:
        filtered_data = filtered_data[filtered_data[feature] <= x_max]
    
    # Настройка стиля
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    if show_boxplot:
        fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] + 2), 
                                  gridspec_kw={'height_ratios': [1, 3]})
        
        # Box plot
        sns.boxplot(x=filtered_data[feature], ax=axes[0], 
                    color="#3498db", linewidth=2)
        axes[0].set_title(f'Box Plot of {feature}', fontsize=14, fontweight='bold', pad=10)
        axes[0].set_xlabel('')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Гистограмма с KDE
        sns.histplot(
            filtered_data[feature], 
            bins=bins, 
            kde=True, 
            color=color, 
            alpha=alpha,
            line_kws={'color': '#e74c3c', 'linewidth': 2.5},
            ax=axes[1]
        )
        axes[1].set_title(f'Distribution of {feature}', fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel(feature, fontsize=13, fontweight='semibold')
        axes[1].set_ylabel('Frequency', fontsize=13, fontweight='semibold')
        
        # Добавление статистики на график
        mean_val = filtered_data[feature].mean()
        median_val = filtered_data[feature].median()
        std_val = filtered_data[feature].std()
        
        axes[1].axvline(mean_val, color="#2e31cc", linestyle='--', 
                        linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[1].axvline(median_val, color="#f31212", linestyle='-.', 
                        linewidth=2, label=f'Median: {median_val:.2f}')
        axes[1].legend(fontsize=11, frameon=True, framealpha=0.9)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        
        sns.histplot(
            filtered_data[feature], 
            bins=bins, 
            kde=True, 
            color=color, 
            alpha=alpha,
            line_kws={'color': '#e74c3c', 'linewidth': 2.5}
        )
        
        plt.title(f'Distribution of {feature}', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel(feature, fontsize=13, fontweight='semibold')
        plt.ylabel('Frequency', fontsize=13, fontweight='semibold')
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        mean_val = filtered_data[feature].mean()
        median_val = filtered_data[feature].median()
        std_val = filtered_data[feature].std()
        
        plt.axvline(mean_val, color='#2ecc71', linestyle='--', 
                    linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='#f39c12', linestyle='-.', 
                    linewidth=2, label=f'Median: {median_val:.2f}')
        plt.legend(fontsize=11, frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    # Вывод статистики
    print(f"Статистика для '{feature}':")
    print(f"  Mean:   {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Std:    {std_val:.2f}")
    print(f"  Min:    {filtered_data[feature].min():.2f}")
    print(f"  Max:    {filtered_data[feature].max():.2f}")
    print(f"  Count:  {len(filtered_data)}")
    if show_boxplot:
        q1 = filtered_data[feature].quantile(0.25)
        q3 = filtered_data[feature].quantile(0.75)
        iqr = q3 - q1
        print(f"  Q1:     {q1:.2f}")
        print(f"  Q3:     {q3:.2f}")
        print(f"  IQR:    {iqr:.2f}")

# столбчатая диаграмма для вывода категориальных признаков
def plot_bar_categorical(data, feature, figsize=(8, 6), top_n=None, 
                         color_palette='viridis', show_percent=True, 
                         rotation=45, order=None):
    """
    Построение столбчатой диаграммы для категориального признака.
    
    Параметры:
    ----------
    data : pd.DataFrame
        Исходный датафрейм
    feature : str
        Название столбца для визуализации
    figsize : tuple
        Размер графика (ширина, высота)
    top_n : int, optional
        Показать только top N категорий по частоте
    color_palette : str or list
        Цветовая палитра для столбцов
    show_percent : bool
        Показывать ли проценты на столбцах
    rotation : int
        Угол поворота подписей категорий
    order : list, optional
        Порядок отображения категорий
    """
    # Настройка стиля
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    plt.figure(figsize=figsize)
    
    # Подсчёт частот
    if top_n is not None:
        value_counts = data[feature].value_counts().head(top_n)
    else:
        value_counts = data[feature].value_counts()
    
    if order is None:
        order = value_counts.index
    
    # Создание столбчатой диаграммы
    ax = sns.countplot(data=data, y=feature, order=order, palette=color_palette)
    
    # Оформление
    plt.title(f'Distribution of {feature}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Count', fontsize=13, fontweight='semibold')
    plt.ylabel(feature, fontsize=13, fontweight='semibold')
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Добавление процентов и значений на столбцы
    if show_percent:
        total = len(data)
        for i, v in enumerate(value_counts.values):
            percent = (v / total) * 100
            ax.text(v + 5, i, f'{v}\n({percent:.1f}%)', 
                    va='center', fontsize=10, fontweight='semibold')
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print(f"Статистика для '{feature}':")
    print(f"  Unique categories: {data[feature].nunique()}")
    print(f"  Total count: {len(data)}")
    print(f"\nTop categories:")
    for cat, count in value_counts.head(10).items():
        percent = (count / len(data)) * 100
        print(f"  {cat}: {count} ({percent:.1f}%)")

# диаграмма-пирог для вывода категориальных признаков
def plot_pie_categorical(data, feature, figsize=(8, 8), top_n=None, 
                         color_palette='Set2', show_percent=True):
    """
    Построение круговой диаграммы для категориального признака.
    
    Параметры:
    ----------
    data : pd.DataFrame
        Исходный датафрейм
    feature : str
        Название столбца для визуализации
    figsize : tuple
        Размер графика
    top_n : int, optional
        Показать только top N категорий, остальные объединить в 'Other'
    color_palette : str or list
        Цветовая палитра
    show_percent : bool
        Показывать ли проценты на секторах
    """
    # Настройка стиля
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Подсчёт частот
    if top_n is not None:
        value_counts = data[feature].value_counts()
        if len(value_counts) > top_n:
            top_categories = value_counts.head(top_n)
            other_count = value_counts.iloc[top_n:].sum()
            top_categories['Other'] = other_count
            value_counts = top_categories
        else:
            value_counts = value_counts
    else:
        value_counts = data[feature].value_counts()
    
    # Круговая диаграмма
    wedges, texts, autotexts = ax.pie(
        value_counts.values, 
        labels=value_counts.index,
        autopct='%1.1f%%' if show_percent else None,
        colors=sns.color_palette(color_palette, len(value_counts)),
        startangle=90,
        pctdistance=0.85,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    
    # Оформление
    ax.set_title(f'Distribution of {feature}', fontsize=16, fontweight='bold', pad=20)
    
    # Настройка шрифта процентов
    if show_percent:
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
    
    # Добавление круга в центре для создания "donut" эффекта
    centre_circle = plt.Circle((0, 0), 0.70, fc='white', linewidth=2)
    ax.add_artist(centre_circle)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print(f"Статистика для '{feature}':")
    print(f"  Unique categories: {data[feature].nunique()}")
    print(f"  Total count: {len(data)}")
    print(f"\nCategory distribution:")
    for cat, count in value_counts.items():
        percent = (count / len(data)) * 100
        print(f"  {cat}: {count} ({percent:.1f}%)")