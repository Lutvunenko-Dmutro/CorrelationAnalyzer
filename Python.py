import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects # Для обводки тексту
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import networkx as nx
import random

# --- ГЕНЕРАТОР ДАНИХ (ACADEMIC DISTRIBUTION) ---
def generate_academic_data(num_samples=200):
    # 1. Дохід (Базова змінна, логнормальний розподіл)
    income = np.random.lognormal(mean=10.5, sigma=0.5, size=num_samples)
    income = np.round(income, -2)
    
    # 2. Витрати (Сильна залежність від Доходу)
    expenses = income * 0.75 + np.random.normal(0, income * 0.05, num_samples)
    expenses = np.maximum(expenses, 1000)

    # 3. Покупки (Залежність від Витрат)
    purchases = expenses / 2000 + np.random.normal(0, 2, num_samples)
    purchases = np.round(np.clip(purchases, 1, 50))

    # 4. Час на сайті (Залежність від Покупок)
    time_on_site = 10 + purchases * 2.5 + np.random.normal(0, 8, num_samples)
    time_on_site = np.round(np.clip(time_on_site, 5, 180))

    # 5. Оцінка (Слабка/Помірна залежність від Часу)
    satisfaction = 3.0 + (time_on_site / 150) + np.random.normal(0, 0.8, num_samples)
    satisfaction = np.round(np.clip(satisfaction, 1, 5), 1)

    # 6. Вік (НЕЗАЛЕЖНА змінна - для створення контрасту на гістограмі)
    age = np.random.normal(40, 12, num_samples)
    age = np.round(np.clip(age, 18, 75))

    # 7. ID (Шум)
    ids = np.random.permutation(np.arange(1000, 1000 + num_samples))

    df = pd.DataFrame({
        'ID_Клієнта': ids,
        'Вік_користувача': age,
        'Річний_Дохід': income,
        'Сума_Витрат': expenses,
        'Кількість_Транзакцій': purchases,
        'Час_активності_хв': time_on_site,
        'Індекс_лояльності': satisfaction
    })

    # Додаємо трохи "сміття" для реалізму
    for col in ['Річний_Дохід', 'Сума_Витрат']:
        indices = np.random.choice(df.index, 3, replace=False)
        df.loc[indices, col] = np.nan

    return df

# --- ОСНОВНА ПРОГРАМА ---
plt.style.use('dark_background')

class CorrelationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Correlation Analysis System v5.2 (Academic Pro)")
        self.root.geometry("1300x850")
        
        self.df_cleaned = None
        self.corr_matrix = None
        self.STRONG_CORR_THRESHOLD = 0.70 # Поріг для відображення на графі

        self.create_menu()
        
        # Стилізація вкладок
        self.notebook = ttk.Notebook(root, bootstyle="dark")
        
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_network = ttk.Frame(self.notebook)
        self.tab_ranking = ttk.Frame(self.notebook)
        self.tab_distribution = ttk.Frame(self.notebook)
        self.tab_interpret = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_heatmap, text='1. Матриця (Heatmap)')
        self.notebook.add(self.tab_network, text='2. Граф (Network)')
        self.notebook.add(self.tab_ranking, text='3. Рейтинг (Ranking)')
        self.notebook.add(self.tab_distribution, text='4. Розподіл (Distribution)')
        self.notebook.add(self.tab_interpret, text='5. Інтерпретація (AI Insights)')
        
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        for tab in [self.tab_heatmap, self.tab_network, self.tab_ranking, self.tab_distribution]:
            self.show_placeholder(tab)
            
        self.setup_interpretation_tab()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="🔄 Згенерувати демо-дані", command=self.load_demo_data)
        file_menu.add_separator()
        file_menu.add_command(label="📂 Завантажити CSV...", command=self.load_csv)
        file_menu.add_command(label="❌ Вихід", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

    def show_placeholder(self, tab, message="Очікування даних...\nВиберіть 'Файл' -> 'Згенерувати демо-дані'"):
        for widget in tab.winfo_children():
            widget.destroy()
        frame = ttk.Frame(tab)
        frame.pack(expand=True, fill='both')
        lbl = ttk.Label(frame, text=message, font=("Segoe UI", 14), bootstyle="secondary")
        lbl.pack(expand=True)

    def load_demo_data(self):
        df = generate_academic_data()
        self.process_data(df)
        messagebox.showinfo("Успіх", "Демо-дані успішно згенеровано!")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                self.process_data(df)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin1')
                    self.process_data(df)
                    messagebox.showinfo("Інформація", "Файл відкрито у кодуванні Latin-1.")
                except Exception as e:
                    messagebox.showerror("Помилка", f"Не вдалося розпізнати кодування файлу.\n{e}")
            except Exception as e:
                messagebox.showerror("Помилка", f"{e}")

    def process_data(self, df_raw):
        rename_map = {
            "Log GDP per capita": "GDP (ВВП)",
            "Healthy life expectancy": "Health (Здоров'я)",
            "Healthy life expectancy at birth": "Health (Здоров'я)",
            "Freedom to make life choices": "Freedom (Свобода)",
            "Ladder score": "Happiness (Щастя)",
            "Life Ladder": "Happiness (Щастя)",
            "Perceptions of corruption": "Corruption (Корупція)",
            "Social support": "Social (Підтримка)",
            "Generosity": "Generosity (Щедрість)"
        }
        df_raw = df_raw.rename(columns=rename_map)
        junk_words = ['whisker', 'residual', 'year', 'regional', 'indicator', 'dystopia']
        cols_to_drop = [c for c in df_raw.columns if any(junk in c.lower() for junk in junk_words)]
        
        if cols_to_drop:
            df_raw = df_raw.drop(columns=cols_to_drop, errors='ignore')
            print(f"Автоматично видалено технічні колонки: {cols_to_drop}")

        # --- ЕТАП 1: СТАНДАРТНА ОБРОБКА ---
        df_numeric = df_raw.select_dtypes(include=np.number)
        self.df_cleaned = df_numeric.dropna()
        
        if self.df_cleaned.shape[1] < 2:
            messagebox.showwarning("Помилка", "Недостатньо числових колонок для аналізу.")
            return

        # --- ЕТАП 2: РОЗРАХУНОК ---
        self.corr_matrix = self.df_cleaned.corr(method='pearson')
        
        # --- ЕТАП 3: ОНОВЛЕННЯ GUI ---
        self.update_heatmap_tab()
        self.update_network_tab()
        self.update_ranking_tab()
        self.update_distribution_tab()
        self.setup_interpretation_tab()
        
        # Повідомляємо користувача, скільки колонок залишилось
        messagebox.showinfo("Аналіз завершено", 
                            f"Завантажено записів: {len(self.df_cleaned)}\n"
                            f"Аналізуємо змінних: {len(self.df_cleaned.columns)}\n\n"
                            f"Ми автоматично прибрали технічні дані (роки, похибки), "
                            f"щоб показати найцікавіше!")

    def clear_tab(self, tab):
        for widget in tab.winfo_children():
            widget.destroy()

    # --- 1. HEATMAP (Всі цифри чорні + фікс відступів) ---
    def update_heatmap_tab(self):
        self.clear_tab(self.tab_heatmap)
        
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        
        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool))
        
        sns.heatmap(self.corr_matrix, mask=mask, annot=True, fmt=".2f", 
                    cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .7},
                    annot_kws={"color": "black", "fontsize": 9, "fontweight": "bold"}) # Чорний жирний текст
        
        ax.set_title("Матриця кореляцій Пірсона", color='white', fontsize=14, pad=15)
        
        # Фікс обрізання тексту знизу
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white', rotation=0)
        
        # Colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        plt.subplots_adjust(bottom=0.25) # ВАЖЛИВО: Відступ знизу

        canvas = FigureCanvasTkAgg(fig, master=self.tab_heatmap)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- 2. NETWORK (Граф, фікс накладання) ---
    def update_network_tab(self):
        self.clear_tab(self.tab_network)
        
        # 1. Створення графа
        G = nx.Graph()
        cols = self.corr_matrix.columns
        
        # Додаємо ребра
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = self.corr_matrix.iloc[i, j]
                if abs(val) >= self.STRONG_CORR_THRESHOLD:
                    G.add_edge(cols[i], cols[j], weight=val)

        if G.number_of_edges() == 0:
            self.show_placeholder(self.tab_network, "Немає сильних зв'язків для побудови графа.")
            return

        # 2. Налаштування полотна
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.set_facecolor('#2b2b2b')
        ax.axis('off')

        # 3. Алгоритм розміщення (Layout)
        # k - відстань між вузлами. Чим менше зв'язків, тим більше k, щоб граф не був "злиплим"
        k_val = 2.0 if len(G.nodes) < 5 else 0.8
        pos = nx.spring_layout(G, seed=42, k=k_val, iterations=50)

        # 4. РОЗМІР ВУЗЛІВ (Залежить від кількості зв'язків - Degree Centrality)
        # Вузли-хаби будуть більшими
        d = dict(G.degree)
        node_sizes = [v * 600 + 1500 for v in d.values()] 

        # 5. КОЛІР ВУЗЛІВ (Градієнт)
        # Використовуємо cmap для красивого кольору
        node_colors = list(d.values())

        # Малюємо вузли
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, 
                                       node_size=node_sizes, 
                                       node_color=node_colors, 
                                       cmap='viridis', # Красива палітра
                                       edgecolors='white', # Білий обідок
                                       linewidths=2)

        # 6. СТИЛІЗАЦІЯ РЕБЕР (Залежить від сили кореляції)
        edges = G.edges(data=True)
        weights = [abs(data['weight']) for u, v, data in edges]
        
        # Колір ребра: Червоний для (+), Синій для (-)
        edge_colors = ['#ff6b6b' if data['weight'] > 0 else '#4ecdc4' for u, v, data in edges]
        
        # Товщина ребра: чим сильніший зв'язок, тим товще (масштабуємо)
        widths = [(w - self.STRONG_CORR_THRESHOLD + 0.1) * 10 for w in weights]

        nx.draw_networkx_edges(G, pos, ax=ax, 
                               width=widths, 
                               edge_color=edge_colors, 
                               alpha=0.7) # Прозорість, щоб бачити перетини

        # 7. ТЕКСТ ВУЗЛІВ (Halo Effect - Білий текст з чорною обводкою)
        for node, (x, y) in pos.items():
            clean_name = node.replace('_', '\n') # Розбиваємо довгі назви на 2 рядки
            t = ax.text(x, y, clean_name, 
                        fontsize=9, 
                        fontweight='bold', 
                        color='white', 
                        ha='center', va='center')
            t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])

        # 8. ПІДПИСИ КОЕФІЦІЄНТІВ (Тільки цифра з обводкою, без жовтих квадратів)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        
        # Малюємо підписи ребер вручну для кращого контролю
        text_items = nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, 
                                     font_color='white', font_size=8, label_pos=0.5, rotate=False)
        
        # Додаємо обводку до цифр на ребрах
        for t in text_items.values():
            t.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

        ax.set_title(f"Топологія сильних зв'язків (|r| > {self.STRONG_CORR_THRESHOLD})", 
                     color='white', fontsize=14, pad=10)

        canvas = FigureCanvasTkAgg(fig, master=self.tab_network)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- 3. RANKING (Назви всередині стовпчиків) ---
    def update_ranking_tab(self):
        self.clear_tab(self.tab_ranking)
        
        # 1. Підготовка даних
        corr_pairs = self.corr_matrix.unstack()
        sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
        unique_pairs = []
        seen = set()
        
        for idx, val in sorted_pairs.items():
            v1, v2 = idx
            if v1 != v2 and (v2, v1) not in seen:
                seen.add((v1, v2))
                unique_pairs.append({'Пара': f"{v1} ↔ {v2}", 'Коефіцієнт': val, 'Абс': abs(val)})
        
        df_pairs = pd.DataFrame(unique_pairs)
        df_top = df_pairs.sort_values(by='Абс', ascending=False).head(10) # Топ-10
        
        if df_top.empty: return

        # 2. Налаштування полотна
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        
        # 3. Малюємо графік
        bar_plot = sns.barplot(data=df_top, x='Коефіцієнт', y='Пара', hue='Пара', legend=False, ax=ax, palette='viridis', edgecolor='white', alpha=0.9)
        
        # Ховаємо осі та зайві рамки
        ax.set_ylabel(None)
        ax.set_yticklabels([]) # Прибираємо старі підписи
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        
        # Встановлюємо межі, щоб було місце для цифр справа
        max_val = df_top['Абс'].max()
        ax.set_xlim(0, max_val * 1.25) # +25% місця справа

        # 4. РЕНДЕРИНГ ТЕКСТУ
        for i, bar in enumerate(ax.patches):
            if i < len(df_top):
                raw_text = df_top.iloc[i]['Пара']
                clean_text = raw_text.replace('_', ' ') 
                
                txt_name = ax.text(
                    x=0.02, 
                    y=bar.get_y() + bar.get_height() / 2, 
                    s=clean_text, 
                    color='white', 
                    ha='left', 
                    va='center', 
                    fontsize=11, 
                    fontweight='bold'
                )
                txt_name.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='#2b2b2b')])

                val = df_top.iloc[i]['Коефіцієнт']
                txt_val = ax.text(
                    x=max_val * 1.22, 
                    y=bar.get_y() + bar.get_height() / 2,
                    s=f"{val:.3f}",
                    color='#00ffcc' if i < 3 else 'white', # Топ-3 підсвічуємо
                    ha='right',
                    va='center',
                    fontsize=11,
                    fontfamily='monospace' 
                )

        ax.set_title("Рейтинг кореляцій (Top-10 Ranking)", color='white', fontsize=14, pad=15)
        ax.set_xlabel("Коефіцієнт Пірсона", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.grid(axis='x', linestyle='--', alpha=0.1) 
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab_ranking)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- 4. DISTRIBUTION (Гістограма + Таблиця) ---
    def update_distribution_tab(self):
        self.clear_tab(self.tab_distribution)
        
        # Розділяємо екран (Верх - 40%, Низ - 60%)
        paned = ttk.PanedWindow(self.tab_distribution, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        frame_top = ttk.Frame(paned)
        frame_bottom = ttk.Frame(paned)
        paned.add(frame_top, weight=4)
        paned.add(frame_bottom, weight=6)
        
        # --- ВЕРХ: ГІСТОГРАМА ---
        values = self.corr_matrix.values.flatten()
        values = values[values != 1.0] 
        
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        
        # Малюємо гістограму
        sns.histplot(values, bins=20, kde=True, color='#17a2b8', ax=ax, edgecolor='white', alpha=0.7)
        
        ax.set_title("Гістограма розподілу сили зв'язків", color='white', fontsize=12)
        ax.set_xlabel("Значення коефіцієнта", color='white')
        ax.set_ylabel("Кількість пар", color='white')
        ax.tick_params(colors='white')
        ax.grid(axis='y', linestyle='--', alpha=0.2)
        
        canvas = FigureCanvasTkAgg(fig, master=frame_top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- НИЗ: SMART LIST (Список пар замість матриці) ---
        
        # 1. Готуємо дані (розгортаємо матрицю в список)
        corr_pairs = self.corr_matrix.unstack()
        sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
        unique_pairs = []
        seen = set()
        
        for idx, val in sorted_pairs.items():
            v1, v2 = idx
            if v1 != v2 and (v2, v1) not in seen: # Прибираємо дублікати і діагональ
                seen.add((v1, v2))
                abs_val = abs(val)
                if abs_val >= 0.7: status = "Сильний"
                elif abs_val >= 0.3: status = "Помірний"
                else: status = "Слабкий"
                
                unique_pairs.append((v1, v2, val, status))
        
        # 2. Створюємо таблицю
        cols = ('Змінна A', 'Змінна B', 'Коефіцієнт r', 'Статус')
        tree = ttk.Treeview(frame_bottom, columns=cols, show='headings', bootstyle="dark")
        
        # Налаштування колонок
        tree.heading('Змінна A', text='Змінна A', anchor=tk.W)
        tree.heading('Змінна B', text='Змінна B', anchor=tk.W)
        tree.heading('Коефіцієнт r', text='Коефіцієнт r', anchor=tk.CENTER)
        tree.heading('Статус', text='Сила зв\'язку', anchor=tk.CENTER)
        
        tree.column('Змінна A', width=200)
        tree.column('Змінна B', width=200)
        tree.column('Коефіцієнт r', width=100, anchor=tk.CENTER)
        tree.column('Статус', width=120, anchor=tk.CENTER)
        
        # Налаштування кольорових тегів
        tree.tag_configure('strong_pos', foreground='#00ff00') 
        tree.tag_configure('strong_neg', foreground='#ff4444') 
        tree.tag_configure('moderate', foreground='#ffcc00')   
        tree.tag_configure('weak', foreground='#888888')       
        
        # 3. Заповнюємо таблицю з розфарбовкою
        for v1, v2, val, stat in unique_pairs:
            # Визначаємо, який колір дати
            tag = 'weak'
            if abs(val) >= 0.7:
                tag = 'strong_pos' if val > 0 else 'strong_neg'
            elif abs(val) >= 0.3:
                tag = 'moderate'
                
            # Вставляємо рядок
            tree.insert("", "end", values=(v1, v2, f"{val:.4f}", stat), tags=(tag,))
            
        # Скролбар
        vsb = ttk.Scrollbar(frame_bottom, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Додаємо біндинг для кліку (щоб працювала інтерпретація)
        tree.bind("<<TreeviewSelect>>", lambda e: self.on_smart_list_select(e, tree))

    # --- ОНОВЛЕНИЙ ОБРОБНИК КЛІКУ ДЛЯ НОВОГО СПИСКУ ---
    def on_smart_list_select(self, event, tree):
        try:
            item = tree.focus()
            if not item: return
            
            vals = tree.item(item)['values']
            # У новому списку порядок: Var1, Var2, Val, Status
            var1 = vals[0]
            var2 = vals[1]
            val = float(vals[2])
            
            self.generate_report(var1, var2, val)
        except Exception as e:
            print(f"Error selecting item: {e}")

    # --- 5. INTERPRETATION ---
    def setup_interpretation_tab(self):
        self.clear_tab(self.tab_interpret)
        
        # Головний контейнер з відступами
        main_frame = ttk.Frame(self.tab_interpret, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. ЗАГОЛОВОК (Header)
        self.lbl_header = ttk.Label(
            main_frame, 
            text="Виберіть пару змінних у списку для аналізу", 
            font=("Segoe UI", 18, "bold"), 
            bootstyle="inverse-light"
        )
        self.lbl_header.pack(pady=(0, 20), anchor="center")
        
        # 2. ВІЗУАЛЬНИЙ БЛОК (Спідометр + Картки)
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.X, pady=10)
        
        # --- Спідометр (Meter) ---
        # Це "фішка" ttkbootstrap - виглядає дуже модерново
        self.meter = ttk.Meter(
            viz_frame,
            metersize=220,
            padding=5,
            amountused=0,
            metertype="semi",       # Півколо
            subtext="Сила зв'язку",
            interactive=False,      # Користувач не може крутити
            textright="%",
            bootstyle="success",
            stripethickness=10
        )
        self.meter.pack(side=tk.LEFT, padx=40)
        
        # --- Картки з метриками (Stats Cards) ---
        stats_frame = ttk.Frame(viz_frame)
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Картка 1: Коефіцієнт
        self.card_r = ttk.Labelframe(stats_frame, text="Коефіцієнт Пірсона (r)", bootstyle="info", padding=15)
        self.card_r.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.lbl_r_val = ttk.Label(self.card_r, text="--", font=("Consolas", 24, "bold"), foreground="#17a2b8")
        self.lbl_r_val.pack()
        
        # Картка 2: R-квадрат (Детермінація)
        self.card_r2 = ttk.Labelframe(stats_frame, text="R² (Вплив)", bootstyle="warning", padding=15)
        self.card_r2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        self.lbl_r2_val = ttk.Label(self.card_r2, text="--%", font=("Consolas", 24, "bold"), foreground="#ffc107")
        self.lbl_r2_val.pack()
        ttk.Label(self.card_r2, text="спільна варіація", font=("Segoe UI", 9)).pack()

        # 3. ТЕКСТОВИЙ БЛОК (Insights) - Гарно оформлений
        self.insight_frame = ttk.Labelframe(main_frame, text="🤖 AI Insights (Висновки)", bootstyle="light", padding=15)
        self.insight_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        self.lbl_insight_title = ttk.Label(self.insight_frame, text="", font=("Segoe UI", 14, "bold"))
        self.lbl_insight_title.pack(anchor="w", pady=(0, 10))
        
        self.lbl_insight_body = ttk.Label(self.insight_frame, text="Очікування даних...", font=("Segoe UI", 12), wraplength=1100)
        self.lbl_insight_body.pack(anchor="w")
        
        # Початковий стан - ховаємо віджети поки немає вибору
        viz_frame.pack_forget()
        self.insight_frame.pack_forget()
        
        # Зберігаємо посилання на фрейми, щоб потім їх показати
        self.viz_container = viz_frame

    def on_list_select(self, event, tree):
        try:
            item = tree.focus()
            if not item: return
            vals = tree.item(item)['values']
            var1 = vals[0]
            col_id = tree.identify_column(event.x)
            col_idx = int(col_id.replace('#', '')) - 1
            if col_idx > 0:
                var2 = tree.heading(col_id)['text']
                val = float(vals[col_idx])
                self.generate_report(var1, var2, val)
        except: pass

    def generate_report(self, v1, v2, r):
        self.notebook.select(self.tab_interpret)
        
        # Показуємо сховані елементи
        self.viz_container.pack(fill=tk.X, pady=10)
        self.insight_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 1. Оновлюємо заголовок
        self.lbl_header.config(text=f"{v1}  ↔  {v2}")
        
        # 2. Оновлюємо Спідометр та Картки
        abs_r = abs(r)
        self.meter.configure(amountused=int(abs_r * 100))
        self.lbl_r_val.config(text=f"{r:.3f}")
        self.lbl_r2_val.config(text=f"{r**2 * 100:.1f}%")
        
        # 3. Визначаємо кольори та тексти
        sign = "прямий (+)" if r > 0 else "зворотний (-)"
        
        if abs_r >= 0.9:
            status = "ДУЖЕ СИЛЬНИЙ"
            bootstyle = "success" # Зелений
            meaning = "Ці показники майже ідентичні у своїй динаміці. Зміна одного гарантує зміну іншого."
            action = "✅ Можна сміливо використовувати один показник для прогнозування іншого."
        elif abs_r >= 0.7:
            status = "СИЛЬНИЙ"
            bootstyle = "success"
            meaning = "Існує чітка залежність. Вони, ймовірно, є частиною одного процесу."
            action = "✅ Варто враховувати цей зв'язок при плануванні стратегії."
        elif abs_r >= 0.4:
            status = "ПОМІРНИЙ"
            bootstyle = "warning" # Жовтий
            meaning = "Зв'язок є, але на нього впливають інші фактори (шум)."
            action = "⚠️ Використовувати обережно. Потрібен додатковий аналіз."
        else:
            status = "СЛАБКИЙ"
            bootstyle = "secondary" # Сірий
            meaning = "Показники змінюються хаотично відносно один одного."
            action = "❌ Не витрачайте час на пошук закономірностей."

        # Оновлюємо колір спідометра
        self.meter.configure(bootstyle=bootstyle)

        # --- НОВИЙ КРАСИВИЙ ВИВІД ТЕКСТУ ---
        
        # Очищаємо старий текст
        self.lbl_insight_title.config(text="") 
        self.lbl_insight_body.config(text="")

        # Формуємо структуру звіту (використовуємо форматування)
        report_text = f"""
 СТАТУС ЗВ'ЯЗКУ: {status} {sign}
 ─────────────────────────────────────────────
 
 💡 ЩО ЦЕ ОЗНАЧАЄ:
 {meaning}
 
 ─────────────────────────────────────────────
 
 🚀 РЕКОМЕНДАЦІЯ:
 {action}
        """
        
        # Оновлюємо текст і колір заголовка рамки
        self.insight_frame.configure(text=f" Висновок AI системи ", bootstyle=bootstyle)
        self.lbl_insight_body.config(text=report_text, font=("Consolas", 11))

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = CorrelationApp(root)
    root.mainloop()
