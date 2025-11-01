import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class CorrelationApp:
    """
    Головний клас програми для кореляційного аналізу.
    (Версія 3.0 з інтерпретацією результатів)
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Інструмент для кореляційного аналізу (v3.0 з Інтерпретацією)")
        self.root.geometry("900x700") 

        self.df_cleaned = None
        self.corr_matrix = None
        self.tree_grouping = None 
        self.tree_list = None     
        
        self.STRONG_CORR_THRESHOLD = 0.8
        plt.style.use('dark_background')
        self.create_menu()

        # --- Створення системи вкладок ---
        self.notebook = ttk.Notebook(root, bootstyle="dark")
        
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_grouping = ttk.Frame(self.notebook)
        self.tab_list = ttk.Frame(self.notebook)
        self.tab_interpret = ttk.Frame(self.notebook) 

        self.notebook.add(self.tab_heatmap, text='Теплова карта (Heatmap)')
        self.notebook.add(self.tab_grouping, text='Групування (r > 0.8)')
        self.notebook.add(self.tab_list, text='Список коефіцієнтів')
        self.notebook.add(self.tab_interpret, text='Інтерпретація 💡')
        
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.setup_interpretation_tab()
        self.show_welcome_message(self.tab_heatmap)
        self.show_welcome_message(self.tab_grouping)
        self.show_welcome_message(self.tab_list)

    def create_menu(self):
        """Створює головне меню програми."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Завантажити CSV...", command=self.load_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Вихід", command=self.root.quit)
        
        menubar.add_cascade(label="Файл", menu=file_menu)

    def show_welcome_message(self, tab, message=None):
        """Показує вітальне повідомлення до завантаження даних."""
        for widget in tab.winfo_children():
            widget.destroy()
            
        if not message:
            message = "Будь ласка, завантажте CSV-файл через меню 'Файл' -> 'Завантажити CSV...'"
            
        welcome_label = ttk.Label(
            tab,
            text=message,
            font=("Arial", 12),
            justify=tk.CENTER,
            padding=20,
            bootstyle="secondary"
        )
        welcome_label.pack(expand=True)

    def setup_interpretation_tab(self):
        """Налаштовує вміст вкладки "Інтерпретація"."""
        self.clear_tab(self.tab_interpret)
        
        info_label = ttk.Label(
            self.tab_interpret,
            text="Клікніть на будь-який рядок у вкладках 'Групування' або 'Список коефіцієнтів', щоб побачити пояснення тут.",
            font=("Arial", 10, "italic"),
            bootstyle="info",
            padding=(10, 10)
        )
        info_label.pack(side=tk.TOP, fill=tk.X)
        
        # Використовуємо ScrolledText для прокрутки, якщо пояснення довге
        self.interpret_text_area = scrolledtext.ScrolledText(
            self.tab_interpret,
            wrap=tk.WORD,
            font=("Arial", 12),
            height=10,
            bg="#303030", 
            fg="white",   
            padx=15,
            pady=15,
            relief=tk.FLAT
        )
        self.interpret_text_area.pack(expand=True, fill='both', padx=10, pady=(0, 10))
        self.interpret_text_area.insert(tk.END, "Очікую на вибір...")
        self.interpret_text_area.config(state=tk.DISABLED)

    def load_csv(self):
        """
        Відкриває діалог вибору файлу та запускає процес
        завантаження, обробки та аналізу.
        """
        file_path = filedialog.askopenfilename(
            title="Оберіть CSV-файл",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return

        try:
            df_raw = pd.read_csv(file_path)
            self.process_data(df_raw)
            
        except Exception as e:
            messagebox.showerror("Помилка завантаження", f"Не вдалося прочитати файл:\n{e}")

    def process_data(self, df_raw):
        """
        Обробляє завантажений DataFrame:
        1. Очищує дані (лише числові, без NaN).
        2. Розраховує кореляційну матрицю.
        3. Оновлює всі вкладки з результатами.
        """
        
        df_numeric = df_raw.select_dtypes(include=np.number)
        original_cols = len(df_numeric.columns)
        original_rows = len(df_numeric)
        
        self.df_cleaned = df_numeric.dropna()
        cleaned_rows = len(self.df_cleaned)
        
        if self.df_cleaned.empty or len(self.df_cleaned.columns) < 2:
            messagebox.showwarning(
                "Помилка даних",
                "Після очищення (видалення пропусків та нечислових колонок) не залишилося достатньо даних для аналізу."
            )
            return

        self.corr_matrix = self.df_cleaned.corr(method='pearson')
        
        self.update_heatmap_tab()
        self.update_list_tab()
        self.update_grouping_tab()
        self.setup_interpretation_tab() 
        
        dropped_rows = original_rows - cleaned_rows
        dropped_cols = len(df_raw.columns) - original_cols
        
        info_msg = (
            f"Завантажено: {original_rows} записів.\n"
            f"Видалено нечислових колонок: {dropped_cols}.\n"
            f"Видалено записів з пропусками: {dropped_rows}.\n"
            f"Залишилося для аналізу: {cleaned_rows} записів."
        )
        messagebox.showinfo("Обробка завершена", info_msg)

    def clear_tab(self, tab):
        """Очищує вкладку від старих віджетів."""
        for widget in tab.winfo_children():
            widget.destroy()

    def update_heatmap_tab(self):
        """Оновлює вкладку "Теплова карта"."""
        self.clear_tab(self.tab_heatmap)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.set_facecolor('#222222')
        ax.set_facecolor('#222222')

        sns.heatmap(
            self.corr_matrix,
            annot=True,     
            fmt='.2f',      
            cmap='coolwarm',
            linewidths=.5,
            ax=ax,
            cbar_kws={"label": "Шкала кореляції"}
        )
        ax.set_title("Теплова карта коефіцієнтів кореляції Пірсона", color='white')
        ax.tick_params(colors='white')
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')
        ax.figure.axes[-1].yaxis.label.set_color('white')
        ax.figure.axes[-1].tick_params(colors='white')
        
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.tab_heatmap)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_list_tab(self):
        """Оновлює вкладку "Список коефіцієнтів" у вигляді таблиці."""
        self.clear_tab(self.tab_list)

        frame = ttk.Frame(self.tab_list)
        frame.pack(expand=True, fill='both')

        cols = ['Змінна'] + list(self.corr_matrix.columns)
        self.tree_list = ttk.Treeview(frame, columns=cols, show='headings', bootstyle="darkly")
        
        for col in cols:
            self.tree_list.heading(col, text=col)
            self.tree_list.column(col, width=100, anchor=tk.CENTER)
            
        for index, row in self.corr_matrix.iterrows():
            values = [index] + [f"{val:.3f}" for val in row]
            self.tree_list.insert("", "end", values=values)
            
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_list.yview, bootstyle="secondary round")
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree_list.xview, bootstyle="secondary round")
        self.tree_list.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.tree_list.pack(expand=True, fill='both')
        
        self.tree_list.bind("<<TreeviewSelect>>", self.on_list_select)

    def update_grouping_tab(self):
        """Оновлює вкладку "Групування"."""
        self.clear_tab(self.tab_grouping)
        
        frame = ttk.Frame(self.tab_grouping)
        frame.pack(expand=True, fill='both')

        cols = ("Змінна 1", "Змінна 2", "Коефіцієнт")
        self.tree_grouping = ttk.Treeview(frame, columns=cols, show='headings', bootstyle="darkly")
        
        for col in cols:
            self.tree_grouping.heading(col, text=col)
            self.tree_grouping.column(col, width=150, anchor=tk.CENTER)

        pairs = []
        for i in range(len(self.corr_matrix.columns)):
            for j in range(i + 1, len(self.corr_matrix.columns)):
                col1 = self.corr_matrix.columns[i]
                col2 = self.corr_matrix.columns[j]
                corr_val = self.corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= self.STRONG_CORR_THRESHOLD:
                    pairs.append((col1, col2, f"{corr_val:.3f}"))
        
        pairs.sort(key=lambda x: abs(float(x[2])), reverse=True)
        
        for pair in pairs:
            self.tree_grouping.insert("", "end", values=pair)
            
        self.tree_grouping.pack(expand=True, fill='both', side='left')
        
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_grouping.yview, bootstyle="secondary round")
        vsb.pack(side='right', fill='y')
        self.tree_grouping.configure(yscrollcommand=vsb.set)
        
        self.tree_grouping.bind("<<TreeviewSelect>>", self.on_grouping_select)


    def on_grouping_select(self, event):
        """Обробляє клік на таблиці 'Групування'."""
        try:
            selected_item = self.tree_grouping.focus()
            if not selected_item:
                return
                
            item_values = self.tree_grouping.item(selected_item)['values']
            var1, var2, corr_str = item_values
            corr_val = float(corr_str)
            
            self.display_interpretation(var1, var2, corr_val)
        except Exception as e:
            print(f"Помилка інтерпретації (grouping): {e}")

    def on_list_select(self, event):
        """Обробляє клік на таблиці 'Список коефіцієнтів'."""
        try:
            selected_item = self.tree_list.focus()
            if not selected_item:
                return
            
            item = self.tree_list.item(selected_item)
            var1 = item['values'][0] 
            
            column_id = self.tree_list.identify_column(event.x)
            column_index = int(column_id.replace('#', '')) - 1 
            
            if column_index == 0: 
                self.display_interpretation(var1, var1, 1.0)
                return
                
            var2 = self.tree_list.heading(column_id)['text']
            corr_str = item['values'][column_index]
            corr_val = float(corr_str)
            
            self.display_interpretation(var1, var2, corr_val)
        except Exception as e:
            print(f"Помилка інтерпретації (list): {e}")

    def display_interpretation(self, var1, var2, corr_val):
        """Формує текст і показує його на вкладці 'Інтерпретація'."""
        
        interpretation_text = self.interpret_correlation(var1, var2, corr_val)
        
        self.interpret_text_area.config(state=tk.NORMAL)
        self.interpret_text_area.delete(1.0, tk.END)
        self.interpret_text_area.insert(tk.END, interpretation_text)
        self.interpret_text_area.config(state=tk.DISABLED)
        
        self.notebook.select(self.tab_interpret)

    def interpret_correlation(self, var1, var2, corr_val):
        """Генерує текстове пояснення коефіцієнта кореляції."""
        
        if var1 == var2:
            return (
                f"**Інтерпретація зв'язку:**\n\n"
                f"**Змінні:** `{var1}` та `{var2}`\n"
                f"**Коефіцієнт:** `{corr_val:.3f}`\n\n"
                f"**Пояснення:**\n"
                f"Це кореляція змінної самої з собою. Вона завжди дорівнює 1.0 і не несе практичного сенсу, "
                f"окрім перевірки, що дані на місці."
            )

        strength = ""
        direction = ""
        explanation = ""
        
        abs_val = abs(corr_val)
        
        # Визначення сили
        if abs_val >= 0.9:
            strength = "дуже сильний"
        elif abs_val >= 0.7:
            strength = "сильний"
        elif abs_val >= 0.5:
            strength = "середній"
        elif abs_val >= 0.3:
            strength = "слабкий"
        else:
            strength = "дуже слабкий або відсутній"

        # Визначення напрямку
        if corr_val > 0.3:
            direction = "позитивний"
            explanation = f"Коли `{var1}` зростає, `{var2}` також має тенденцію до зростання. І навпаки."
        elif corr_val < -0.3:
            direction = "негативний"
            explanation = f"Коли `{var1}` зростає, `{var2}` має тенденцію до зменшення. І навпаки."
        else:
            direction = "лінійний"
            explanation = f"Між `{var1}` та `{var2}` не спостерігається значущого лінійного зв'язку."
        
        # Формування тексту
        final_text = (
            f"**Інтерпретація зв'язку:**\n\n"
            f"**Змінні:** `{var1}` та `{var2}`\n"
            f"**Коефіцієнт:** `{corr_val:.3f}`\n\n"
            f"**Пояснення:**\n"
            f"Це **{strength} {direction} зв'язок**.\n\n"
            f"**Простими словами:**\n{explanation}\n\n"
            f"--------------------------------------------------\n"
            f"**ВАЖЛИВО:** Пам'ятайте, кореляція НЕ означає причинно-наслідковий зв'язок! "
            f"Ми лише бачимо, що ці змінні рухаються узгоджено, але не можемо стверджувати, "
            f"що одна з них *спричиняє* іншу."
        )
        
        return final_text

# --- Запуск програми ---
if __name__ == "__main__":
    main_window = ttk.Window(themename="darkly")
    app = CorrelationApp(main_window)
    main_window.mainloop()


