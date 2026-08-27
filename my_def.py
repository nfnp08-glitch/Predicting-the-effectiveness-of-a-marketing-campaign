import seaborn as sns
import matplotlib.pyplot as plt

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
                    color='#9b59b6', linewidth=2)
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