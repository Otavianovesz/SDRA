import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Any, List, Optional
import logging
from PIL import Image, ImageTk
import fitz  # PyMuPDF
from datetime import datetime

logger = logging.getLogger('srda.ui.components')

class PDFPreviewPanel(ttk.Frame):
    """
    High-performance PDF Preview using PyMuPDF (fitz) and Canvas.
    Supports Zoom, Pan, and Page Navigation.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.doc: Optional[fitz.Document] = None
        self.current_page_idx = 0
        self.zoom_level = 1.0
        self.image_ref = None # Keep reference to avoid GC
        
        self._build_ui()
        
    def _build_ui(self):
        # Toolbar
        self.toolbar = ttk.Frame(self, bootstyle="secondary")
        self.toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(self.toolbar, text="➖", command=lambda: self.zoom(-0.25), width=3).pack(side=tk.LEFT)
        self.lbl_zoom = ttk.Label(self.toolbar, text="100%")
        self.lbl_zoom.pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="➕", command=lambda: self.zoom(0.25), width=3).pack(side=tk.LEFT)
        
        self.lbl_page = ttk.Label(self.toolbar, text="Página 0/0")
        self.lbl_page.pack(side=tk.RIGHT, padx=5)
        
        # Canvas Area
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        
        self.canvas = tk.Canvas(
            self.canvas_frame,
            bg="#2b2b2b", # Dark professional background
            highlightthickness=0,
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set
        )
        
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)
        
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bindings
        self.canvas.bind("<Control-MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        
        self._drag_data = {"x": 0, "y": 0}

    def load_document(self, file_path: str):
        try:
            if self.doc:
                self.doc.close()
            
            self.doc = fitz.open(file_path)
            self.current_page_idx = 0
            self.render()
            self._update_status()
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(
                200, 200, 
                text=f"Erro ao abrir PDF:\n{str(e)}", 
                fill="red", 
                font=("Segoe UI", 12)
            )

    def render(self, overlay_boxes: List[Dict[str, Any]] = None):
        if not self.doc: return
        
        try:
            page = self.doc[self.current_page_idx]
            mat = fitz.Matrix(self.zoom_level, self.zoom_level)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL -> ImageTk
            img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            self.image_ref = ImageTk.PhotoImage(img_data)
            
            self.canvas.delete("all")
            self.canvas.create_image(
                10, 10, # Margin
                anchor=tk.NW,
                image=self.image_ref
            )
            
            # --- V4.0 Feature: OCR Overlay ---
            if overlay_boxes:
                for box in overlay_boxes:
                    try:
                        # Normalize coordinates (assuming box is [x0, y0, x1, y1] unscaled)
                        rect = fitz.Rect(box['bbox'])
                        # Apply current zoom matrix
                        rect_scaled = rect * mat
                        
                        # Apply margin offset
                        x0, y0, x1, y1 = rect_scaled.x0 + 10, rect_scaled.y0 + 10, rect_scaled.x1 + 10, rect_scaled.y1 + 10
                        
                        color = box.get('color', 'red')
                        label = box.get('label', '')
                        
                        # Draw Rect
                        self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2, tags="overlay")
                        
                        # Draw Label Background + Text
                        if label:
                            self.canvas.create_rectangle(x0, y0-15, x0+len(label)*7, y0, fill=color, outline=color, tags="overlay")
                            self.canvas.create_text(x0+2, y0-8, text=label, anchor=tk.W, fill="white", font=("Segoe UI", 8, "bold"), tags="overlay")
                            
                    except Exception as e:
                        logger.warning(f"Failed to draw overlay box: {e}")
            
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
            self.lbl_zoom.config(text=f"{int(self.zoom_level * 100)}%")
            
        except Exception as e:
            logger.error(f"Render error: {e}")

    def zoom(self, delta):
        new_zoom = self.zoom_level + delta
        if 0.25 <= new_zoom <= 5.0:
            self.zoom_level = new_zoom
            self.render()

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self.zoom(0.25)
        else:
            self.zoom(-0.25)
            
    def _on_drag_start(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag(self, event):
        dx = self._drag_data["x"] - event.x
        dy = self._drag_data["y"] - event.y
        self.canvas.xview_scroll(dx // 10, "units") # Smoothing
        self.canvas.yview_scroll(dy // 10, "units")
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y
    
    def _update_status(self):
        if self.doc:
            self.lbl_page.config(text=f"Doc: {self.doc.name} | Pág: {self.current_page_idx + 1}/{len(self.doc)}")


class SmartEditPanel(ttk.Frame):
    """
    Inline Editor Panel. 
    Replaces the modal dialog for a seamless experience.
    """
    def __init__(self, master, db_callback=None, save_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.db = db_callback # Function to get DB instance or DB object
        self.save_callback = save_callback
        self.current_doc_data = None
        self.modified = False
        
        self._build_ui()
        
    def _build_ui(self):
        # Header
        ttk.Label(self, text="Painel de Edição", font=("Segoe UI", 12, "bold"), bootstyle="inverse-primary").pack(fill=tk.X, padx=5, pady=5)
        
        # Form Container
        self.form = ttk.Frame(self, padding=10)
        self.form.pack(fill=tk.BOTH, expand=True)
        
        # Fields
        self.vars = {
            "type": tk.StringVar(),
            "supplier": tk.StringVar(),
            "amount": tk.StringVar(),
            "due_date": tk.StringVar(),
            "payment_date": tk.StringVar()
        }
        
        self._add_field("Tipo Documental", self.vars["type"], is_combo=True, 
                       values=['BOLETO', 'COMPROVANTE', 'NFE', 'NFSE', 'FATURA'])
        self._add_field("Fornecedor", self.vars["supplier"])
        self._add_field("Valor (R$)", self.vars["amount"])
        self._add_field("Vencimento", self.vars["due_date"])
        self._add_field("Pagamento", self.vars["payment_date"])
        
        # Actions
        btn_frame = ttk.Frame(self, padding=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Button(btn_frame, text="Salvar Alterações", bootstyle="success", command=self._save).pack(fill=tk.X)

    def _add_field(self, label, var, is_combo=False, values=None):
        frame = ttk.Frame(self.form)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label, font=("Segoe UI", 9)).pack(anchor=tk.W)
        
        if is_combo:
            cb = ttk.Combobox(frame, textvariable=var, values=values, state="readonly")
            cb.pack(fill=tk.X)
        else:
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(fill=tk.X)
            
    def load_data(self, doc_data: Dict[str, Any]):
        self.current_doc_data = doc_data
        
        self.vars["type"].set(doc_data.get('doc_type', 'UNKNOWN'))
        self.vars["supplier"].set(doc_data.get('supplier_clean', ''))
        
        amt = doc_data.get('amount_cents', 0)
        self.vars["amount"].set(f"{amt/100:.2f}".replace(".", ",") if amt else "")
        
        self.vars["due_date"].set(self._iso_to_br(doc_data.get('due_date', '')))
        self.vars["payment_date"].set(self._iso_to_br(doc_data.get('payment_date', '')))
        
    def _save(self):
        if not self.current_doc_data: return
        
        # Parse Amount
        try:
            amt_str = self.vars["amount"].get().replace(".", "").replace(",", ".")
            amount_cents = int(float(amt_str) * 100) if amt_str else 0
        except:
            amount_cents = 0

        new_data = {
            "doc_type": self.vars["type"].get(),
            "supplier": self.vars["supplier"].get().upper(),
            "amount_cents": amount_cents,
            "due_date": self._br_to_iso(self.vars["due_date"].get()),
            "payment_date": self._br_to_iso(self.vars["payment_date"].get()),
        }
        
        if self.save_callback:
            self.save_callback(self.current_doc_data['id'], new_data)
            
    # Helpers (Date utils duplicate from main, better to move to utils later)
    def _iso_to_br(self, d):
        if not d: return ""
        try: return datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y")
        except: return d

    def _br_to_iso(self, d):
        if not d: return None
        try: return datetime.strptime(d.strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
        except: return None
