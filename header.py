import os
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib
import numpy as np
from scipy.stats import gmean

class CSVColumnSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Preprocessing & Visualization")
        self.root.geometry("1280x900")
        ctk.set_default_color_theme("green")

        # Title
        ctk.CTkLabel(root, text="CSV Preprocessing & Visualization", font=("Helvetica", 36, "bold"), text_color="white").pack(pady=20, padx=10, fill='x')

        # Buttons Frame
        self.frame_buttons = ctk.CTkFrame(root)
        self.frame_buttons.pack(fill='x', padx=20, pady=10)

        ctk.CTkButton(self.frame_buttons, text="Upload CSV", command=self.load_csv, width=200, fg_color="#4CAF50").pack(side="left", padx=15, pady=10)
        ctk.CTkButton(self.frame_buttons, text="Preprocess Data", command=self.process_data, width=200, fg_color="#2196F3").pack(side="left", padx=15, pady=10)
        ctk.CTkButton(self.frame_buttons, text="Exit", command=root.quit, width=200, fg_color="#F44336").pack(side="left", padx=15, pady=10)

        # Frames for tables and navigation
        self.frame_tables = ctk.CTkFrame(root)
        self.frame_tables.pack(fill='both', expand=True, padx=20, pady=10)

        self.df_original = None
        self.df_processed = None

        # Navigation
        self.nav_frame = ctk.CTkFrame(root)
        self.nav_frame.pack(fill='x', padx=20, pady=10)

        ctk.CTkButton(self.nav_frame, text="View Original Data", command=lambda: self.switch_frame(self.frame_table_original), width=200).pack(side="left", padx=10)
        ctk.CTkButton(self.nav_frame, text="View Processed Data", command=lambda: self.switch_frame(self.frame_table_processed), width=200).pack(side="left", padx=10)

        # Data Table Frames
        self.frame_table_original = ctk.CTkFrame(self.frame_tables)
        self.frame_table_processed = ctk.CTkFrame(self.frame_tables)
        self.frame_table_original.pack(fill='both', expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df_original = pd.read_csv(file_path)
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            self.display_table(self.df_original, self.frame_table_original, "Original Data")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")

    def preprocess_data(self, data):
        try:
            # Fill missing values
            for col in data.select_dtypes(include=["float64", "int64"]).columns:
                non_missing_values = data[col].dropna()
                positive_values = non_missing_values[non_missing_values > 0]
                if len(positive_values) > 0:
                    geometric_mean = gmean(positive_values)
                    data.loc[data[col].isna(), col] = geometric_mean

            # Scale numeric data
            scaler = StandardScaler()
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            joblib.dump(scaler, "scaler.pkl")

            # Encode categorical columns
            label_encoders = {}
            for col in data.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le
            joblib.dump(label_encoders, "encoder.pkl")

            # Apply PCA
            pca = PCA(n_components=min(len(numeric_cols), 5))
            pca_data = pca.fit_transform(data[numeric_cols])
            pca_columns = [f"PCA_{i+1}" for i in range(pca_data.shape[1])]
            pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=data.index)
            data.drop(columns=numeric_cols, inplace=True)
            data = pd.concat([data, pca_df], axis=1)
            joblib.dump(pca, "PCA.pkl")

            return data
        except Exception as e:
            messagebox.showerror("Error", f"Error during preprocessing: {e}")
            return None

    def process_data(self):
        if self.df_original is None:
            messagebox.showerror("Error", "No CSV file loaded!")
            return

        self.df_processed = self.preprocess_data(self.df_original.copy())
        if self.df_processed is not None:
            self.display_table(self.df_processed, self.frame_table_processed, "Processed Data")
            self.switch_frame(self.frame_table_processed)

    def display_table(self, df, frame, title):
        for widget in frame.winfo_children():
            widget.destroy()

        ctk.CTkLabel(frame, text=title, font=("Arial", 16, "bold"), text_color="black").pack()
        columns = df.columns.tolist()
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=15)
        tree.pack(fill='both', expand=True, padx=10, pady=10)

        for col in columns:
            tree.heading(col, text=col, anchor="center")
            tree.column(col, anchor="center", width=150)

        for _, row in df.iterrows():
            tree.insert("", "end", values=row.tolist())

    def switch_frame(self, frame):
        self.frame_table_original.pack_forget()
        self.frame_table_processed.pack_forget()
        frame.pack(fill='both', expand=True)

if __name__ == "__main__":
    root = ctk.CTk()
    app = CSVColumnSelector(root)
    root.mainloop()
