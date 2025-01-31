import os
import pandas as pd
import ttkbootstrap as ttk
from tkinter import filedialog, messagebox, StringVar, IntVar
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import numpy as np
from scipy.stats import gmean

class CSVColumnSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Column Selector & Preprocessing")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        self.style = ttk.Style()
        self.style.theme_use("cosmo")

        self.label_title = ttk.Label(root, text="CSV Column Selector & Preprocessing", font=("Arial", 20, "bold"))
        self.label_title.pack(pady=10)

        self.btn_select_file = ttk.Button(root, text="Select CSV File", command=self.load_csv, bootstyle="primary")
        self.btn_select_file.pack(pady=10)

        self.frame_checkbox_container = ttk.Frame(root)
        self.frame_checkbox_container.pack(pady=10, padx=10, fill="both", expand=True)

        self.canvas = ttk.Canvas(self.frame_checkbox_container)
        self.scrollbar = ttk.Scrollbar(self.frame_checkbox_container, orient="vertical", command=self.canvas.yview)
        self.frame_checkboxes = ttk.Frame(self.canvas)

        self.frame_checkboxes.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.frame_checkboxes, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.preprocess_options()
        self.btn_show_selected = ttk.Button(root, text="Preprocess & Show Data", command=self.show_selected_columns, state="disabled", bootstyle="success")
        self.btn_show_selected.pack(pady=10)

        self.text_output = ttk.Text(root, height=20, wrap="none", font=("Arial", 10))
        self.text_output.pack(pady=10, padx=10, fill="both", expand=True)

        self.df = None
        self.checkboxes = {}

    def preprocess_options(self):
        self.options_frame = ttk.Frame(self.root)
        self.options_frame.pack(pady=10)

        self.fill_null = IntVar(value=1)
        self.scaler_val = IntVar(value=1)
        self.apply_pca = IntVar(value=0)

        ttk.Checkbutton(self.options_frame, text="Fill Missing Values", variable=self.fill_null, bootstyle="primary").pack(side="left", padx=10)
        ttk.Checkbutton(self.options_frame, text="Scale Data", variable=self.scaler_val, bootstyle="primary").pack(side="left", padx=10)
        ttk.Checkbutton(self.options_frame, text="Apply PCA", variable=self.apply_pca, bootstyle="primary").pack(side="left", padx=10)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df = pd.read_csv(file_path)
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            for widget in self.frame_checkboxes.winfo_children():
                widget.destroy()

            self.checkboxes.clear()
            for col in self.df.columns:
                var = ttk.BooleanVar()
                chk = ttk.Checkbutton(self.frame_checkboxes, text=col, variable=var, bootstyle="primary")
                chk.pack(anchor="w", padx=10, pady=5, ipadx=10, ipady=5)
                self.checkboxes[col] = var

            self.btn_show_selected.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")

    def preprocess_data(self, data):
        try:
            if self.fill_null.get() == 1:
                for col in data.select_dtypes(include=["float64", "int64"]).columns:
                    non_missing_values = data[col].dropna()
                    positive_values = non_missing_values[non_missing_values > 0]
                    if len(positive_values) > 0:
                        geometric_mean = gmean(positive_values)
                        data.loc[data[col].isna(), col] = geometric_mean

            if self.scaler_val.get() == 1:
                scaler = StandardScaler()
                numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
                data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
                joblib.dump(scaler, "scaler.pkl")

            label_encoders = {}
            for col in data.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le
            joblib.dump(label_encoders, "encoder.pkl")

            if self.apply_pca.get() == 1:
                numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
                pca = PCA(n_components=min(len(numeric_cols), 5))
                pca_data = pca.fit_transform(data[numeric_cols])
                pca_columns = [f"PCA_{i + 1}" for i in range(pca_data.shape[1])]
                pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
                data.drop(columns=numeric_cols, inplace=True)
                data = pd.concat([data, pca_df], axis=1)
                joblib.dump(pca, "PCA.pkl")

            return data
        except Exception as e:
            messagebox.showerror("Error", f"Error during preprocessing: {e}")
            return None

    def show_selected_columns(self):
        processed_data = self.preprocess_data(self.df)
        if processed_data is not None:
            output_text = processed_data.to_string(index=False)
            self.text_output.delete("1.0", "end")
            self.text_output.insert("end", output_text)

if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    app = CSVColumnSelector(root)
    root.mainloop()
