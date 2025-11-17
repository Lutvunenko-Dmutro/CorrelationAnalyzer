import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import networkx as nx

class CorrelationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Інструмент для кореляційного аналізу (v4.1 Final)")
        self.root.geometry("1200x800") 

        self.df_cleaned = None
        self.corr_matrix = None
        self.STRONG_CORR_THRESHOLD = 0.8
        plt.style.use('dark_background')
        self.create_menu()

        self.notebook = ttk.Notebook(root, bootstyle="dark")
        
        self.tab_heatmap = ttk.Frame(self.notebook)
        self.tab_graph = ttk.Frame(self.notebook)
        self.tab_grouping = ttk.Frame(self.notebook)
        self.tab_list = ttk.Frame(self.notebook)
        self.tab_interpret = ttk.Frame(self.notebook) 

        self.notebook.add(self.tab_heatmap, text='Теплова карта')
        self.notebook.add(self.tab_graph, text='Граф та Рейтинг 📊')
        self.notebook.add(self.tab_grouping, text=f'Групування (r > {self.STRONG_CORR_THRESHOLD})')
        self.notebook.add(self.tab_list, text='Список коефіцієнтів')
        self.notebook.add(self.tab_interpret, text='Інтерпретація 💡')
        
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.setup_interpretation_tab()
        self.show_welcome_message(self.tab_heatmap)
        self.show_welcome_message(self.tab_graph)
        self.show_welcome_message(self.tab_grouping)
        self.show_welcome_message(self.tab_list)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Завантажити CSV...", command=self.load_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Вихід", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

    def show_welcome_message(self, tab, message=None):
        for widget in tab.winfo_children():
            widget.destroy()
        if not message:
            message = "Завантажте файл через меню 'Файл' -> 'Завантажити CSV...'"
        ttk.Label(tab, text=message, font=("Arial", 12), justify=tk.CENTER, bootstyle="secondary").pack(expand=True)

    def setup_interpretation_tab(self):
        self.clear_tab(self.tab_interpret)
        ttk.Label(self.tab_interpret, text="Клікніть на рядок у таблицях для пояснення.", font=("Arial", 10, "italic"), bootstyle="info").pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.interpret_text_area = scrolledtext.ScrolledText(self.tab_interpret, wrap=tk.WORD, font=("Arial", 12), height=10, bg="#303030", fg="white", padx=15, pady=15, relief=tk.FLAT)
        self.interpret_text_area.pack(expand=True, fill='both', padx=10, pady=(0, 10))
        self.interpret_text_area.insert(tk.END, "Очікую на вибір...")
        self.interpret_text_area.config(state=tk.DISABLED)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.process_data(pd.read_csv(file_path))
            except Exception as e:
                messagebox.showerror("Помилка", f"{e}")

    def process_data(self, df_raw):
        df_numeric = df_raw.select_dtypes(include=np.number)
        self.df_cleaned = df_numeric.dropna()
        
        if self.df_cleaned.empty or len(self.df_cleaned.columns) < 2:
            messagebox.showwarning("Помилка", "Недостатньо даних.")
            return

        self.corr_matrix = self.df_cleaned.corr(method='pearson')
        
        self.update_heatmap_tab()
        self.update_graph_tab() # Оновлено
        self.update_list_tab()
        self.update_grouping_tab()
        self.setup_interpretation_tab() 
        
        messagebox.showinfo("Успіх", f"Оброблено {len(self.df_cleaned)} записів.")

    def clear_tab(self, tab):
        for widget in tab.winfo_children():
            widget.destroy()

    def update_heatmap_tab(self):
        self.clear_tab(self.tab_heatmap)
        plt.close('all')
        
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.set_facecolor('#222222')
        ax.set_facecolor('#222222')

        sns.heatmap(self.corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={"label": "Коефіцієнт"})
        
        ax.set_title("Матриця кореляцій", color='white', fontsize=12)
        ax.tick_params(colors='white')
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.tab_heatmap)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_graph_tab(self):
        """Будує ДВА графіки: Мережу (Graph) та Стовпчики (Bar Chart)."""
        self.clear_tab(self.tab_graph)
        
        G = nx.Graph()
        cols = self.corr_matrix.columns
        pairs_data = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = self.corr_matrix.iloc[i, j]
                if abs(val) >= self.STRONG_CORR_THRESHOLD:
                    G.add_edge(cols[i], cols[j], weight=abs(val), label=f"{val:.2f}")
                    pairs_data.append({"Пара": f"{cols[i]}\n{cols[j]}", "Коефіцієнт": val})

        if G.number_of_edges() == 0:
            self.show_welcome_message(self.tab_graph, "Немає сильних зв'язків.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.set_facecolor('#222222')
        
        # --- ЛІВА ЧАСТИНА: МЕРЕЖЕВИЙ ГРАФ ---
        ax1.set_facecolor('#222222')
        pos = nx.spring_layout(G, seed=42, k=0.5) 
        
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='#00bc8c', node_size=2800, edgecolors='white')
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='white', width=2, alpha=0.6)
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=9, font_weight='bold', font_color='white')
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, ax=ax1, edge_labels=edge_labels, font_color='black', font_size=8, rotate=False)
        
        ax1.set_title("Структура зв'язків", color='white', fontsize=14)
        ax1.axis('off')
        ax1.margins(0.15) 

        # --- ПРАВА ЧАСТИНА: СТОВПЧИКОВА ДІАГРАМА ---
        ax2.set_facecolor('#222222')
        df_pairs = pd.DataFrame(pairs_data).sort_values(by="Коефіцієнт", ascending=False).head(10) # Топ-10
        
        sns.barplot(x="Коефіцієнт", y="Пара", hue="Пара", data=df_pairs, ax=ax2, palette="dark:g", edgecolor="white", legend=False)
        
        ax2.set_title("Топ найсильніших зв'язків", color='white', fontsize=14)
        ax2.set_xlabel("Коефіцієнт Пірсона", color='white')
        ax2.set_ylabel("")
        ax2.tick_params(colors='white', labelsize=9)
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        ax2.set_xlim(0.8, 1.01) 
        
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.3f', padding=3, color='white')

        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab_graph)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_list_tab(self):
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
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_list.yview)
        vsb.pack(side='right', fill='y')
        self.tree_list.configure(yscrollcommand=vsb.set)
        self.tree_list.pack(expand=True, fill='both')
        self.tree_list.bind("<<TreeviewSelect>>", self.on_list_select)

    def update_grouping_tab(self):
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
                val = self.corr_matrix.iloc[i, j]
                if abs(val) >= self.STRONG_CORR_THRESHOLD:
                    pairs.append((self.corr_matrix.columns[i], self.corr_matrix.columns[j], f"{val:.3f}"))
        pairs.sort(key=lambda x: abs(float(x[2])), reverse=True)
        
        for pair in pairs:
            self.tree_grouping.insert("", "end", values=pair)
        
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree_grouping.yview)
        vsb.pack(side='right', fill='y')
        self.tree_grouping.configure(yscrollcommand=vsb.set)
        self.tree_grouping.pack(expand=True, fill='both')
        self.tree_grouping.bind("<<TreeviewSelect>>", self.on_grouping_select)

    def on_grouping_select(self, event):
        try:
            item = self.tree_grouping.focus()
            if item:
                vals = self.tree_grouping.item(item)['values']
                self.display_interpretation(vals[0], vals[1], float(vals[2]))
        except: pass

    def on_list_select(self, event):
        try:
            item = self.tree_list.focus()
            if item:
                var1 = self.tree_list.item(item)['values'][0]
                col_id = self.tree_list.identify_column(event.x)
                idx = int(col_id.replace('#', '')) - 1
                if idx > 0:
                    var2 = self.tree_list.heading(col_id)['text']
                    val = float(self.tree_list.item(item)['values'][idx])
                    self.display_interpretation(var1, var2, val)
        except: pass

    def display_interpretation(self, var1, var2, corr_val):
        text = self.interpret_correlation(var1, var2, corr_val)
        self.interpret_text_area.config(state=tk.NORMAL)
        self.interpret_text_area.delete(1.0, tk.END)
        self.interpret_text_area.insert(tk.END, text)
        self.interpret_text_area.config(state=tk.DISABLED)
        self.notebook.select(self.tab_interpret)

    def interpret_correlation(self, var1, var2, corr_val):
        if var1 == var2: return "Кореляція змінної самої з собою = 1.0"
        abs_val = abs(corr_val)
        strength = "дуже сильний" if abs_val >= 0.9 else "сильний" if abs_val >= 0.7 else "середній" if abs_val >= 0.5 else "слабкий"
        direction = "позитивний" if corr_val > 0 else "негативний"
        return f"**{var1} - {var2}**\nКоефіцієнт: {corr_val}\nЦе {strength} {direction} зв'язок."

if __name__ == "__main__":
    main_window = ttk.Window(themename="darkly")
    app = CorrelationApp(main_window)
    main_window.mainloop()
