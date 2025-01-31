import os
import pandas as pd
import ttkbootstrap as ttk
import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk as tk_tt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np
from scipy.stats import gmean

# Configure CustomTkinter appearance
ctk.set_appearance_mode("light")  # "light", "dark", "system"
ctk.set_default_color_theme("blue")  # Other available themes: "green", "dark-blue"

class CSVProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Data Processor - Classification")
        self.root.geometry("1280x900")
        self.root.minsize(1000, 700)

        # Using ttkbootstrap theme
        self.style = ttk.Style()
        self.style.theme_use("cosmo")

        # Data storage
        self.df_original = None
        self.df_processed = None

        # Create pages
        self.create_pages()
        # Create navigation bar
        self.create_nav_bar()

        # Show Upload Page first
        self.show_page(self.page_upload)

    def create_nav_bar(self):
        """Top navigation bar for easy page switching."""
        self.nav_bar = ctk.CTkFrame(self.root, height=50)
        self.nav_bar.pack(side="top", fill="x", padx=10, pady=(10, 0))

        btn_style = {"width": 180, "corner_radius": 6, "fg_color": "#3B8ED0"}

        ctk.CTkButton(
            self.nav_bar, text="Upload CSV", command=lambda: self.show_page(self.page_upload), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="View Data", command=lambda: self.show_page(self.page_data_view), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Select Columns", command=lambda: self.show_page(self.page_column_select), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Results", command=lambda: self.show_page(self.page_results), **btn_style
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            self.nav_bar, text="Run Classification", command=lambda: self.show_page(self.page_model_select), **btn_style
        ).pack(side="left", padx=5)

    def create_pages(self):
        """Initialize all pages."""
        self.page_upload = ctk.CTkFrame(self.root)
        self.page_data_view = ctk.CTkFrame(self.root)
        self.page_column_select = ctk.CTkFrame(self.root)
        self.page_results = ctk.CTkFrame(self.root)
        self.page_model_select = ctk.CTkFrame(self.root)
        self.page_model_results = ctk.CTkFrame(self.root)

        self.create_upload_page()
        self.create_data_view_page()
        self.create_column_selection_page()
        self.create_results_page()
        self.create_model_selection_page()
        self.create_model_results_page()

    def create_upload_page(self):
        """Page 1: Upload CSV File"""
        for widget in self.page_upload.winfo_children():
            widget.destroy()
        self.page_upload.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_upload,
            text="Upload CSV File",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=30, padx=20)

        upload_btn = ctk.CTkButton(
            self.page_upload,
            text="Select File",
            command=self.load_csv,
            width=300,
            fg_color="#4CAF50",
            font=("Helvetica", 16)
        )
        upload_btn.grid(row=1, column=0, pady=20)

        self.page_upload.pack(fill="both", expand=True)

    def create_data_view_page(self):
        """Page 2: Display Uploaded CSV Data"""
        for widget in self.page_data_view.winfo_children():
            widget.destroy()
        self.page_data_view.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_data_view,
            text="Uploaded CSV Data",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        self.frame_table_original = ctk.CTkFrame(self.page_data_view)
        self.frame_table_original.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_data_view.rowconfigure(1, weight=1)

        next_btn = ctk.CTkButton(
            self.page_data_view,
            text="Next: Select Label and Features",
            command=self.show_page_column_select,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        next_btn.grid(row=2, column=0, pady=20)

    def create_column_selection_page(self):
        """Page 3: Select Label and Feature Columns"""
        for widget in self.page_column_select.winfo_children():
            widget.destroy()
        self.page_column_select.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_column_select,
            text="Select Label and Features",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        # Frame for label selection
        self.label_frame = ctk.CTkFrame(self.page_column_select)
        self.label_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.label_frame.grid_columnconfigure(0, weight=1)

        # Frame for feature selection
        self.column_frame = ctk.CTkFrame(self.page_column_select)
        self.column_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        self.page_column_select.rowconfigure(2, weight=1)

        self.label_var = ttk.StringVar()

        # Create label selection dropdown
        label_label = ctk.CTkLabel(
            self.label_frame,
            text="Select Label Column:",
            font=("Arial", 18, "bold")
        )
        label_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")

        self.label_menu = ttk.Combobox(
            self.label_frame,
            state="readonly",
            textvariable=self.label_var,
            font=("Arial", 14)
        )
        self.label_menu.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.label_menu.bind("<<ComboboxSelected>>", self.update_feature_selection)

        # Process Data button
        process_btn = ctk.CTkButton(
            self.page_column_select,
            text="Process Data",
            command=self.process_data,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        process_btn.grid(row=3, column=0, pady=20)

    def create_results_page(self):
        """Page 4: Show Processed & Original Data with Navigation Options"""
        for widget in self.page_results.winfo_children():
            widget.destroy()
        self.page_results.grid_columnconfigure(0, weight=1)

        nav_frame = ctk.CTkFrame(self.page_results)
        nav_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        nav_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_original = ctk.CTkButton(
            nav_frame,
            text="View Original Data",
            command=lambda: self.show_table(self.df_original, "Original Data"),
            width=200,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_original.grid(row=0, column=0, padx=10)

        self.btn_processed = ctk.CTkButton(
            nav_frame,
            text="View Processed Data",
            command=lambda: self.show_table(self.df_processed, "Processed Data"),
            width=200,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_processed.grid(row=0, column=1, padx=10)

        # Button to jump to classification model page
        self.btn_model = ctk.CTkButton(
            nav_frame,
            text="Run Classification",
            command=lambda: self.show_page(self.page_model_select),
            width=250,
            state="disabled",
            font=("Helvetica", 14)
        )
        self.btn_model.grid(row=0, column=2, padx=10)

        self.result_table_frame = ctk.CTkFrame(self.page_results)
        self.result_table_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_results.rowconfigure(1, weight=1)

    def create_model_selection_page(self):
        """Page 5: Classification Model Processing Page"""
        for widget in self.page_model_select.winfo_children():
            widget.destroy()
        self.page_model_select.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_model_select,
            text="Classification Model",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        info = ctk.CTkLabel(
            self.page_model_select,
            text="Click the button below to train a logistic regression classifier on the processed data.\nA classification report will be generated.",
            font=("Helvetica", 16)
        )
        info.grid(row=1, column=0, pady=10, padx=20)

        run_model_btn = ctk.CTkButton(
            self.page_model_select,
            text="Run Classification",
            command=self.run_classification,
            width=300,
            fg_color="#2196F3",
            font=("Helvetica", 16)
        )
        run_model_btn.grid(row=2, column=0, pady=20)

        back_btn = ctk.CTkButton(
            self.page_model_select,
            text="Back to Results",
            command=lambda: self.show_page(self.page_results),
            width=300,
            font=("Helvetica", 16)
        )
        back_btn.grid(row=3, column=0, pady=10)

    def create_model_results_page(self):
        """Page 6: Display Classification Model Results"""
        for widget in self.page_model_results.winfo_children():
            widget.destroy()
        self.page_model_results.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            self.page_model_results,
            text="Classification Report",
            font=("Helvetica", 32, "bold")
        )
        title.grid(row=0, column=0, pady=20, padx=20)

        self.model_results_text = ctk.CTkTextbox(
            self.page_model_results,
            width=1000,
            height=500,
            font=("Courier", 14)
        )
        self.model_results_text.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.page_model_results.rowconfigure(1, weight=1)

        back_btn = ctk.CTkButton(
            self.page_model_results,
            text="Back to Classification",
            command=lambda: self.show_page(self.page_model_select),
            width=300,
            font=("Helvetica", 16)
        )
        back_btn.grid(row=2, column=0, pady=10)

    def show_page(self, page):
        """Hide all pages and display the selected page."""
        for p in [
            self.page_upload, self.page_data_view, self.page_column_select,
            self.page_results, self.page_model_select, self.page_model_results
        ]:
            p.pack_forget()
            p.grid_forget()
        page.pack(fill="both", expand=True)

    def load_csv(self):
        """Load CSV file and display its contents."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            self.df_original = pd.read_csv(file_path)
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            self.display_table(self.df_original, self.frame_table_original, "Original Data")
            self.show_page(self.page_data_view)
            # Populate the label dropdown options
            self.label_menu['values'] = list(self.df_original.columns)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")

    def show_page_column_select(self, event=None):
        """Populate label and feature selection widgets."""
        for widget in self.label_frame.winfo_children():
            widget.destroy()
        for widget in self.column_frame.winfo_children():
            widget.destroy()

        # Recreate label selection dropdown
        label_label = ctk.CTkLabel(
            self.label_frame,
            text="Select Label Column:",
            font=("Arial", 18, "bold")
        )
        label_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")
        self.label_menu = ttk.Combobox(
            self.label_frame,
            state="readonly",
            textvariable=self.label_var,
            font=("Arial", 14),
            values=list(self.df_original.columns)
        )
        self.label_menu.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.label_menu.bind("<<ComboboxSelected>>", self.update_feature_selection)

        # Create feature selection checkboxes
        self.update_feature_selection()

        self.show_page(self.page_column_select)

    def update_feature_selection(self, event=None):
        """Update feature selection based on chosen label."""
        label_column = self.label_var.get()
        for widget in self.column_frame.winfo_children():
            widget.destroy()

        self.feature_vars = []
        remaining_columns = [col for col in self.df_original.columns if col != label_column]

        for idx, col in enumerate(remaining_columns):
            var = ttk.IntVar()
            chk = ttk.Checkbutton(
                self.column_frame,
                text=col,
                variable=var
            )
            chk.grid(row=idx, column=0, sticky="w", padx=20, pady=5)
            self.feature_vars.append((col, var))

    def process_data(self):
        """Process selected label and features and automatically save preprocessed data."""
        label_column = self.label_var.get()
        selected_features = [col for col, var in self.feature_vars if var.get() == 1]

        if not label_column or not selected_features:
            messagebox.showerror("Error", "Please select a label column and at least one feature!")
            return

        try:
            data_subset = self.df_original[selected_features + [label_column]].copy()
            self.df_processed = self.preprocess_data(data_subset)
            self.btn_original.configure(state="normal")
            self.btn_processed.configure(state="normal")
            self.btn_model.configure(state="normal")
            self.df_processed.to_csv("preprocessed_data.csv", index=False)
            messagebox.showinfo("Download", "Preprocessed data saved as 'preprocessed_data.csv'.")
            self.show_page(self.page_results)
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {e}")

    def preprocess_data(self, data):
        """Preprocessing: fill missing values, standardize numeric columns, encode categoricals."""
        try:
            # Fill missing numeric values with geometric mean (if possible)
            for col in data.select_dtypes(include=["float64", "int64"]).columns:
                non_missing = data[col].dropna()
                positive = non_missing[non_missing > 0]
                if not positive.empty:
                    geom_mean = gmean(positive)
                    data.loc[data[col].isna(), col] = geom_mean

            # Standardize numeric columns
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

            return data
        except Exception as e:
            messagebox.showerror("Error", f"Error during preprocessing: {e}")
            return None

    def display_table(self, df, frame, title):
        """Display a data table inside the provided frame."""
        for widget in frame.winfo_children():
            widget.destroy()

        title_label = ctk.CTkLabel(frame, text=title, font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        container = ctk.CTkFrame(frame)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        tree = tk_tt.Treeview(container, columns=df.columns.tolist(), show="headings")
        vsb = tk_tt.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = tk_tt.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        for col in df.columns:
            tree.heading(col, text=col, anchor="center")
            tree.column(col, anchor="center", width=150)

        for _, row in df.iterrows():
            tree.insert("", "end", values=row.tolist())

    def show_table(self, df, title):
        """Show a data table in the Results page."""
        self.display_table(df, self.result_table_frame, title)

    def run_classification(self):
        """Run a logistic regression classifier on the processed data and display the classification report."""
        label_column = self.label_var.get()
        if label_column not in self.df_processed.columns:
            messagebox.showerror("Error", "Label column not found in processed data.")
            return

        try:
            # Separate features and target
            X = self.df_processed.drop(columns=[label_column])
            y = self.df_processed[label_column]

            # Split data (80/20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train logistic regression classifier
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Generate classification report
            report = classification_report(y_test, y_pred)
            result_text = f"Classification Report:\n\n{report}"

            # Optionally, save the trained model
            joblib.dump(clf, "classification_model.pkl")

            self.show_model_results(result_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error during classification: {e}")

    def show_model_results(self, result_text):
        """Display the classification report on the Model Results page."""
        self.model_results_text.delete("1.0", "end")
        self.model_results_text.insert("end", result_text)
        self.show_page(self.page_model_results)


if __name__ == "__main__":
    root = ctk.CTk()
    app = CSVProcessorApp(root)
    root.mainloop()
