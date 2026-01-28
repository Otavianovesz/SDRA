"""
SRDA-Rural - Interface Grafica Principal v3.0
==============================================
Sistema de Reconciliacao Documental e Automacao Rural

NOVA ARQUITETURA (Vol I, Secao 7):
- Drag-and-Drop nativo via tkinterdnd2
- Visualizacao de grafo de transacoes
- Edicao manual de vinculos
- Hierarquia de datas (Comprovante = MASTER_DATE)

Este modulo implementa a interface interventora, nao apenas visualizadora.
"""

import os
import sys
import shutil
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import queue
import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk as ttk_native  # Native ttk for PanedWindow

# Suporte a Drag-and-Drop
try:
    import tkinterdnd2 as dnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    logging.warning("tkinterdnd2 nao disponivel - drag-drop desabilitado")

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
# Fixed imports: use modern path directly
from ttkbootstrap.widgets.scrolled import ScrolledFrame
from ttkbootstrap.dialogs import Messagebox
try:
    from ttkbootstrap.widgets import ToolTip
except ImportError:
    from ttkbootstrap.tooltip import ToolTip
from tkinter import filedialog
import fitz  # PyMuPDF
from PIL import Image, ImageTk

# Importa modulos do sistema

from database import (
    SRDADatabase,
    DocumentType,
    DocumentStatus,
    EntityTag,
    MatchType
)
from scanner import CognitiveScanner
from mcmf_reconciler import MCMFReconciler, TransactionIsland, DocumentNode
from renamer import DocumentRenamer
from group_matcher import GroupMatcher, DocumentGroup

# Gmail integration (optional)
try:
    from integrations.gmail_connector import GmailConnector, EmailMessage
    GMAIL_AVAILABLE = True
except ImportError:
    GmailConnector = None
    EmailMessage = None
    GMAIL_AVAILABLE = False
    logging.warning("GmailConnector not available")

# Link extractor for deep link hunting
try:
    from integrations.link_extractor import LinkExtractor, download_link_sync
    LINK_EXTRACTOR_AVAILABLE = True
except ImportError:
    LinkExtractor = None
    download_link_sync = None
    LINK_EXTRACTOR_AVAILABLE = False

# Configuracao de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACOES
# ==============================================================================

APP_TITLE = "SRDA-Rural v3.0 - Reconciliacao Cognitiva"
THEME = "darkly"

# Cores de status
STATUS_COLORS = {
    "INGESTED": "info",
    "PARSED": "primary",
    "RECONCILED": "success",
    "RENAMED": "warning",
    "ERROR": "danger",
}

# Cores para tipos de documento
TYPE_COLORS = {
    "NFE": "#4CAF50",
    "NFSE": "#8BC34A",
    "BOLETO": "#2196F3",
    "COMPROVANTE": "#9C27B0",
    "UNKNOWN": "#757575",
}

# Icones de tipo de documento
DOC_ICONS = {
    "NFE": "📄",
    "NFSE": "🛠️",
    "BOLETO": "💰",
    "COMPROVANTE": "✅",
    "UNKNOWN": "❓",
}


# ==============================================================================
# TOAST NOTIFICATION (Premium QoL)
# ==============================================================================

class ToastNotification(tk.Toplevel):
    """
    Non-intrusive notification that auto-dismisses.
    
    Usage:
        ToastNotification(root, "✅ Arquivo processado!", 'success')
    """
    
    STYLES = {
        'success': {'bg': '#27ae60', 'fg': 'white'},
        'error': {'bg': '#e74c3c', 'fg': 'white'},
        'warning': {'bg': '#f39c12', 'fg': 'white'},
        'info': {'bg': '#3498db', 'fg': 'white'},
    }
    
    def __init__(self, parent, message: str, style: str = 'info', duration: int = 3000):
        super().__init__(parent)
        
        # Remove window decorations
        self.overrideredirect(True)
        self.attributes('-topmost', True)
        
        # Set opacity for modern look
        try:
            self.attributes('-alpha', 0.95)
        except:
            pass
        
        # Style
        colors = self.STYLES.get(style, self.STYLES['info'])
        
        # Content frame
        frame = tk.Frame(self, bg=colors['bg'], padx=20, pady=12)
        frame.pack(fill=BOTH, expand=YES)
        
        # Message with icon
        icons = {'success': '✅', 'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}
        icon = icons.get(style, 'ℹ️')
        
        tk.Label(
            frame,
            text=f"{icon} {message}",
            font=('Segoe UI', 11),
            bg=colors['bg'],
            fg=colors['fg']
        ).pack()
        
        # Position in bottom-right corner of parent
        self.update_idletasks()
        w = self.winfo_width()
        h = self.winfo_height()
        px = parent.winfo_x()
        py = parent.winfo_y()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        
        x = px + pw - w - 20
        y = py + ph - h - 60
        self.geometry(f"+{x}+{y}")
        
        # Fade in effect
        self._fade_in()
        
        # Auto-dismiss after duration
        self.after(duration, self._fade_out)
    
    def _fade_in(self):
        """Animate fade in."""
        try:
            alpha = 0.0
            for i in range(10):
                alpha += 0.1
                self.after(i * 30, lambda a=alpha: self.attributes('-alpha', a))
        except:
            pass
    
    def _fade_out(self):
        """Animate fade out and destroy."""
        try:
            alpha = 0.95
            for i in range(10):
                alpha -= 0.1
                self.after(i * 30, lambda a=alpha: self.attributes('-alpha', max(0, a)))
            self.after(350, self.destroy)
        except:
            self.destroy()


# ==============================================================================
# DOCUMENT CONTEXT MENU (Right-Click Actions)
# ==============================================================================

class DocumentContextMenu:
    """
    Context menu for document operations.
    
    Usage:
        menu = DocumentContextMenu(app)
        tree.bind('<Button-3>', menu.show)
    """
    
    def __init__(self, app):
        self.app = app
        self.menu = tk.Menu(app.root, tearoff=0)
        
        # Actions with icons and shortcuts
        self.menu.add_command(
            label="📂 Abrir Arquivo",
            command=app._on_open_file_click,
            accelerator="Ctrl+O"
        )
        self.menu.add_command(
            label="📁 Abrir Pasta",
            command=app._on_open_folder_click
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="✏️ Editar",
            command=app._on_edit_click,
            accelerator="Ctrl+E"
        )
        self.menu.add_command(
            label="🔄 Reprocessar",
            command=app._on_reprocess_click,
            accelerator="Ctrl+R"
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="🔗 Reconciliar",
            command=app._on_match_click
        )
        self.menu.add_command(
            label="📋 Renomear",
            command=app._on_rename_click
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="🗑️ Excluir",
            command=app._on_delete_click,
            accelerator="Delete"
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="📋 Copiar Info",
            command=self._copy_info
        )
    
    def show(self, event):
        """Display context menu at cursor position."""
        # Select the item under cursor
        item = self.app.tree.identify_row(event.y)
        if item:
            self.app.tree.selection_set(item)
            self.app.tree.focus(item)
        
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
    
    def _copy_info(self):
        """Copy selected document info to clipboard."""
        if not self.app.selected_doc_id:
            return
        
        try:
            cursor = self.app.db.connection.cursor()
            cursor.execute("""
                SELECT d.doc_type, t.amount_cents, t.supplier_clean, t.due_date
                FROM documentos d LEFT JOIN transacoes t ON d.id = t.doc_id
                WHERE d.id = ?
            """, (self.app.selected_doc_id,))
            row = cursor.fetchone()
            
            if row:
                doc_type, amount, supplier, due_date = row
                amount_str = SRDADatabase.cents_to_display(amount) if amount else "N/A"
                info = f"{doc_type} | {supplier or 'N/A'} | {amount_str} | Venc: {due_date or 'N/A'}"
                
                self.app.root.clipboard_clear()
                self.app.root.clipboard_append(info)
                self.app._show_toast("📋 Informações copiadas!", 'success')
        except Exception as e:
            self.app._show_toast(f"Erro ao copiar: {e}", 'error')


# ==============================================================================
# QUICK SEARCH WIDGET (Premium Search Bar)
# ==============================================================================

class QuickSearchWidget(ttk.Frame):
    """
    Premium search bar with real-time filtering.
    
    Features:
    - Debounced search (waits for typing to stop)
    - Clear button
    - Placeholder text
    - Keyboard navigation
    """
    
    def __init__(self, parent, on_search=None, placeholder="🔍 Buscar..."):
        super().__init__(parent)
        
        self.on_search = on_search
        self.placeholder = placeholder
        self._search_timer = None
        
        # Search entry
        self.var = tk.StringVar()
        self.var.trace_add('write', self._on_change)
        
        self.entry = ttk.Entry(
            self,
            textvariable=self.var,
            font=('Segoe UI', 10),
            width=25
        )
        self.entry.pack(side=LEFT, fill=X, expand=YES)
        
        # Placeholder
        self._show_placeholder()
        self.entry.bind('<FocusIn>', self._on_focus_in)
        self.entry.bind('<FocusOut>', self._on_focus_out)
        self.entry.bind('<Return>', lambda e: self._do_search())
        self.entry.bind('<Escape>', lambda e: self.clear())
        
        # Clear button
        self.btn_clear = ttk.Button(
            self,
            text="✕",
            width=3,
            bootstyle='secondary-outline',
            command=self.clear
        )
        self.btn_clear.pack(side=LEFT, padx=(5, 0))
        self.btn_clear.pack_forget()  # Hidden initially
    
    def _show_placeholder(self):
        if not self.var.get():
            self.entry.insert(0, self.placeholder)
            self.entry.configure(foreground='gray')
    
    def _on_focus_in(self, event):
        if self.entry.get() == self.placeholder:
            self.entry.delete(0, END)
            self.entry.configure(foreground='')
    
    def _on_focus_out(self, event):
        if not self.var.get():
            self._show_placeholder()
    
    def _on_change(self, *args):
        # Show/hide clear button
        if self.var.get() and self.var.get() != self.placeholder:
            self.btn_clear.pack(side=LEFT, padx=(5, 0))
        else:
            self.btn_clear.pack_forget()
        
        # Debounced search (300ms)
        if self._search_timer:
            self.after_cancel(self._search_timer)
        self._search_timer = self.after(300, self._do_search)
    
    def _do_search(self):
        query = self.var.get()
        if query == self.placeholder:
            query = ""
        if self.on_search:
            self.on_search(query)
    
    def clear(self):
        self.entry.delete(0, END)
        self.entry.configure(foreground='')
        self.btn_clear.pack_forget()
        if self.on_search:
            self.on_search("")
        self.entry.focus_set()
    
    def get(self) -> str:
        val = self.var.get()
        return "" if val == self.placeholder else val


# ==============================================================================
# EMAIL CARD (Sprint 6 - Visual Dashboard)
# ==============================================================================

class EmailCard(ttk.Frame):
    """
    Card visual para email na triagem.
    
    Layout:
    [Status Icon] [Content Icons] [Summary from AI] [Actions]
    🟢            📄 PDF | 🔗 Link   "Agro Baggio - R$ 2.843,29"
    """
    
    STATUS_CONFIG = {
        'ready': {'icon': '🟢', 'style': 'success', 'text': 'Pronto'},
        'warning': {'icon': '🟡', 'style': 'warning', 'text': 'Atenção'},
        'error': {'icon': '🔴', 'style': 'danger', 'text': 'Erro'},
        'pending': {'icon': '🔵', 'style': 'info', 'text': 'Pendente'},
        'scheduled': {'icon': '⚠️', 'style': 'warning', 'text': 'Agendamento'},
    }
    
    CONTENT_ICONS = {
        'pdf': '📄',
        'xml': '⚙️',
        'link': '🔗',
        'attachment': '📎',
        'image': '🖼️',
    }
    
    def __init__(self, parent, email_data: Dict[str, Any], on_process=None, on_view=None):
        """
        Creates an email card.
        
        Args:
            parent: Parent widget
            email_data: Dict with keys:
                - subject: Email subject
                - sender: Sender email
                - date: Date string
                - status: 'ready', 'warning', 'error', 'pending', 'scheduled'
                - content_types: List of 'pdf', 'xml', 'link', 'attachment'
                - summary: AI extracted summary (supplier, value, etc)
                - is_scheduled: If True, shows scheduling warning
            on_process: Callback when process button clicked
            on_view: Callback when view button clicked
        """
        # Determine style based on status
        status = email_data.get('status', 'pending')
        if email_data.get('is_scheduled'):
            status = 'scheduled'
        
        config = self.STATUS_CONFIG.get(status, self.STATUS_CONFIG['pending'])
        
        super().__init__(parent, bootstyle=config['style'], padding=8)
        
        self.email_data = email_data
        self.on_process = on_process
        self.on_view = on_view
        
        self._build_ui(config)
    
    def _build_ui(self, config: Dict):
        """Build the card UI."""
        # Main horizontal layout
        self.columnconfigure(1, weight=1)
        
        # Column 0: Status Icon
        status_label = ttk.Label(
            self,
            text=config['icon'],
            font=('Segoe UI Emoji', 16)
        )
        status_label.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky=W)
        
        # Column 1: Content (subject, details)
        content_frame = ttk.Frame(self)
        content_frame.grid(row=0, column=1, sticky=EW)
        content_frame.columnconfigure(0, weight=1)
        
        # Subject line with content icons
        subject_frame = ttk.Frame(content_frame)
        subject_frame.pack(fill=X)
        
        # Content type icons
        content_types = self.email_data.get('content_types', [])
        icons_text = ' '.join([self.CONTENT_ICONS.get(ct, '') for ct in content_types])
        if icons_text:
            ttk.Label(
                subject_frame,
                text=icons_text,
                font=('Segoe UI Emoji', 10)
            ).pack(side=LEFT, padx=(0, 5))
        
        # Subject
        subject = self.email_data.get('subject', 'Sem assunto')[:50]
        ttk.Label(
            subject_frame,
            text=subject,
            font=('Helvetica', 10, 'bold')
        ).pack(side=LEFT, fill=X, expand=YES)
        
        # Details line (sender, date, summary)
        details_frame = ttk.Frame(content_frame)
        details_frame.pack(fill=X, pady=(2, 0))
        
        sender = self.email_data.get('sender', '')[:30]
        date = self.email_data.get('date', '')
        ttk.Label(
            details_frame,
            text=f"{sender} • {date}",
            font=('Helvetica', 9),
            bootstyle='secondary'
        ).pack(side=LEFT)
        
        # AI Summary (if available)
        summary = self.email_data.get('summary', '')
        if summary:
            ttk.Label(
                details_frame,
                text=f" | {summary[:40]}",
                font=('Helvetica', 9, 'italic'),
                bootstyle='info'
            ).pack(side=LEFT)
        
        # Scheduling warning
        if self.email_data.get('is_scheduled'):
            ttk.Label(
                content_frame,
                text="⚠️ Possível Agendamento - Verificar antes de processar",
                font=('Helvetica', 9, 'bold'),
                bootstyle='warning'
            ).pack(fill=X, pady=(3, 0))
        
        # Column 2: Action buttons
        action_frame = ttk.Frame(self)
        action_frame.grid(row=0, column=2, rowspan=2, padx=(10, 0), sticky=E)
        
        if self.on_view:
            ttk.Button(
                action_frame,
                text="👁",
                command=self._on_view_click,
                bootstyle='info-outline',
                width=3
            ).pack(side=LEFT, padx=2)
        
        if self.on_process:
            ttk.Button(
                action_frame,
                text="▶",
                command=self._on_process_click,
                bootstyle='success-outline',
                width=3
            ).pack(side=LEFT, padx=2)
    
    def _on_view_click(self):
        if self.on_view:
            self.on_view(self.email_data)
    
    def _on_process_click(self):
        if self.on_process:
            self.on_process(self.email_data)


# ==============================================================================
# DIALOG DE EDICAO
# ==============================================================================

class EditDialog(tk.Toplevel):
    """Dialog para edicao de dados de um documento com Active Learning."""
    
    def __init__(self, parent, doc_data: Dict[str, Any], db=None):
        super().__init__(parent)
        self.result = None
        self.doc_data = doc_data
        self.db = db  # For Active Learning logging
        
        # Store original values for correction tracking
        self._original_values = {
            'supplier': doc_data.get('supplier_clean', '') or '',
            'amount_cents': doc_data.get('amount_cents', 0) or 0,
            'due_date': doc_data.get('due_date', ''),
            'payment_date': doc_data.get('payment_date', '')
        }
        
        self.title("Editar Documento")
        self.geometry("450x400")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self._center()
        self._build_ui()
    
    def _center(self):
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() // 2) - 225
        y = self.master.winfo_y() + (self.master.winfo_height() // 2) - 200
        self.geometry(f"+{x}+{y}")
    
    def _build_ui(self):
        main = ttk.Frame(self, padding=20)
        main.pack(fill=BOTH, expand=YES)
        
        ttk.Label(main, text="Editar Dados Extraidos", font=("Helvetica", 14, "bold")).pack(anchor=W, pady=(0, 15))
        
        # Fornecedor
        ttk.Label(main, text="Fornecedor:").pack(anchor=W)
        self.entry_supplier = ttk.Entry(main, width=50)
        self.entry_supplier.pack(fill=X, pady=(2, 10))
        self.entry_supplier.insert(0, self.doc_data.get('supplier_clean', '') or '')
        
        # Valor
        ttk.Label(main, text="Valor (R$):").pack(anchor=W)
        self.entry_amount = ttk.Entry(main, width=20)
        self.entry_amount.pack(anchor=W, pady=(2, 10))
        amount = self.doc_data.get('amount_cents', 0)
        if amount:
            self.entry_amount.insert(0, f"{amount/100:.2f}".replace(".", ","))
        
        # Data de Vencimento
        ttk.Label(main, text="Data de Vencimento (DD/MM/AAAA):").pack(anchor=W)
        self.entry_due_date = ttk.Entry(main, width=20)
        self.entry_due_date.pack(anchor=W, pady=(2, 10))
        if self.doc_data.get('due_date'):
            self.entry_due_date.insert(0, self._iso_to_br(self.doc_data['due_date']))
        
        # Data de Pagamento
        ttk.Label(main, text="Data de Pagamento (DD/MM/AAAA):").pack(anchor=W)
        self.entry_payment_date = ttk.Entry(main, width=20)
        self.entry_payment_date.pack(anchor=W, pady=(2, 10))
        if self.doc_data.get('payment_date'):
            self.entry_payment_date.insert(0, self._iso_to_br(self.doc_data['payment_date']))
        
        # Botoes
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=X, pady=(20, 0))
        
        ttk.Button(btn_frame, text="Cancelar", bootstyle="secondary", command=self.destroy).pack(side=RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Salvar", bootstyle="success", command=self._on_save).pack(side=RIGHT)
    
    def _iso_to_br(self, date_str: str) -> str:
        try:
            parts = date_str.split("-")
            return f"{parts[2]}/{parts[1]}/{parts[0]}"
        except:
            return date_str
    
    def _br_to_iso(self, date_br: str) -> Optional[str]:
        if not date_br:
            return None
        try:
            parts = date_br.split("/")
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        except:
            return None
    
    def _on_save(self):
        amount_str = self.entry_amount.get().replace(".", "").replace(",", ".")
        try:
            amount_cents = int(float(amount_str) * 100) if amount_str else 0
        except:
            amount_cents = 0
        
        new_values = {
            'supplier': self.entry_supplier.get().strip().upper(),
            'amount_cents': amount_cents,
            'due_date': self._br_to_iso(self.entry_due_date.get()),
            'payment_date': self._br_to_iso(self.entry_payment_date.get()),
        }
        
        # Active Learning: Log corrections if values changed
        if self.db and self.doc_data.get('doc_id'):
            doc_id = self.doc_data['doc_id']
            
            # Check each field for changes
            if new_values['supplier'] != self._original_values['supplier']:
                self.db.log_correction(
                    doc_id=doc_id,
                    field_name='fornecedor',
                    ocr_value=self._original_values['supplier'],
                    user_value=new_values['supplier'],
                    original_confidence=self.doc_data.get('field_confidence', {}).get('fornecedor'),
                    extractor_source=self.doc_data.get('extraction_sources', {}).get('fornecedor')
                )
                logger.info(f"[ACTIVE LEARNING] Correção de fornecedor registrada: {self._original_values['supplier']} -> {new_values['supplier']}")
            
            if new_values['amount_cents'] != self._original_values['amount_cents']:
                self.db.log_correction(
                    doc_id=doc_id,
                    field_name='valor',
                    ocr_value=str(self._original_values['amount_cents']),
                    user_value=str(new_values['amount_cents']),
                    original_confidence=self.doc_data.get('field_confidence', {}).get('valor'),
                    extractor_source=self.doc_data.get('extraction_sources', {}).get('valor')
                )
                logger.info(f"[ACTIVE LEARNING] Correção de valor registrada: {self._original_values['amount_cents']} -> {new_values['amount_cents']}")
            
            if new_values['due_date'] and new_values['due_date'] != self._original_values['due_date']:
                self.db.log_correction(
                    doc_id=doc_id,
                    field_name='data_vencimento',
                    ocr_value=self._original_values['due_date'],
                    user_value=new_values['due_date'],
                    original_confidence=self.doc_data.get('field_confidence', {}).get('data_vencimento'),
                    extractor_source=self.doc_data.get('extraction_sources', {}).get('data_vencimento')
                )
        
        self.result = new_values
        self.destroy()


# ==============================================================================
# CLASSE PRINCIPAL: SRDAApplication
# ==============================================================================

class SRDAApplication:
    """
    Aplicacao principal SRDA-Rural v3.0.
    
    Novidades:
    - Drag-and-Drop real via tkinterdnd2
    - Visualizacao de ilhas de transacao (grafo)
    - Vinculacao manual por arraste
    - Hierarquia de datas com Comprovante como MASTER
    """
    
    @staticmethod
    def _iso_to_br(date_str: str) -> str:
        """Helper para formatar data (YYYY-MM-DD -> DD/MM/YYYY)."""
        if not date_str: return ""
        try:
            parts = date_str.split("-")
            return f"{parts[2]}/{parts[1]}/{parts[0]}"
        except:
            return date_str

    def __init__(self):

        # Componentes do sistema
        self.db = SRDADatabase()
        self.scanner = CognitiveScanner(db=self.db)
        self.reconciler = MCMFReconciler(db=self.db)
        self.renamer = DocumentRenamer(db=self.db)
        self.group_matcher = GroupMatcher(db=self.db)  # Sprint 4: Conciliation
        
        # Janela principal (com ou sem DnD)
        if DND_AVAILABLE:
            self.root = dnd.Tk()
            self.root.drop_target_register(dnd.DND_FILES)
            self.root.dnd_bind('<<Drop>>', self._on_external_drop)
            logger.info("Drag-and-Drop habilitado")
        else:
            self.root = tk.Tk()
            logger.warning("Drag-and-Drop desabilitado")
        
        # Aplica tema
        self.style = ttk.Style(theme=THEME)
        
        self.root.title(APP_TITLE)
        self.root.geometry("1400x800")
        self.root.minsize(1100, 650)
        
        self._center_window()
        
        # Estado
        self.selected_doc_id: Optional[int] = None
        self.is_processing = False
        self.preview_image = None
        self.dragged_item = None
        self.current_islands: List[TransactionIsland] = []
        
        # Threading state (MUST be before _build_ui to avoid AttributeError)
        self._email_thread = None
        self._monitor_thread = None
        self._email_pipeline = None
        self._gemini_total_tokens = 0
        self._email_cards = []
        
        # UI
        self._build_ui()
        self._setup_internal_drag_drop()
        self._setup_keyboard_shortcuts()  # QoL: Keyboard shortcuts
        
        # UI Queue for thread safety
        self.gui_queue = queue.Queue()
        self._process_queue()
        
        # Dados iniciais
        self._refresh_all()
    
    def _process_queue(self):
        """Processa eventos da UI enviados por threads (Step 49)."""
        try:
            while True:
                # Tenta pegar evento sem bloquear
                event = self.gui_queue.get_nowait()
                
                # event = (type, data)
                etype, data = event
                
                if etype == "status":
                    self._set_status(data)
                elif etype == "tree_update":
                    # data = (doc_id, status_text)
                    self._update_tree_item_status(data[0], data[1])
                elif etype == "refresh_all":
                    self._refresh_all()
                elif etype == "msg_box":
                    # data = (title, message, type)
                    title, msg, mtype = data
                    if mtype == "info":
                        Messagebox.showinfo(title, msg)
                    elif mtype == "error":
                        Messagebox.show_error(title, msg)
                
                self.gui_queue.task_done()
        except queue.Empty:
            pass
        finally:
            # Reagendar verificação
            self.root.after(100, self._process_queue)

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")
    
    # ==========================================================================
    # DRAG & DROP
    # ==========================================================================
    
    def _on_external_drop(self, event):
        """Handler para arquivos dropados de fora da aplicacao."""
        try:
            files = self.root.tk.splitlist(event.data)
            
            # Validação: Apenas PDFs (Step 52)
            pdf_files = []
            for f in files:
                path = Path(f)
                if path.is_file() and path.suffix.lower() == '.pdf':
                    pdf_files.append(f)
            
            if pdf_files:
                logger.info(f"Recebidos {len(pdf_files)} arquivos via drag-drop")
                for f in pdf_files:
                    self.scanner.process_file(Path(f))
                
                self._refresh_all()
                self._set_status(f"{len(pdf_files)} arquivo(s) processado(s)")
        except Exception as e:
            logger.error(f"Erro no drop externo: {e}")
    
    def _setup_internal_drag_drop(self):
        """Configura drag-drop interno entre itens da TreeView."""
        self.tree.bind("<ButtonPress-1>", self._on_drag_start)
        self.tree.bind("<B1-Motion>", self._on_drag_motion)
        self.tree.bind("<ButtonRelease-1>", self._on_drag_end)
    
    def _on_drag_start(self, event):
        """Inicio do arraste."""
        item = self.tree.identify_row(event.y)
        if item:
            self.dragged_item = item
            self.tree.selection_set(item)
    
    def _on_drag_motion(self, event):
        """Durante o arraste."""
        if self.dragged_item:
            target = self.tree.identify_row(event.y)
            if target and target != self.dragged_item:
                # Visual feedback
                self.tree.selection_set(target)
    
    def _on_drag_end(self, event):
        """Fim do arraste - cria vinculo manual."""
        if not self.dragged_item:
            return
        
        target = self.tree.identify_row(event.y)
        
        if target and target != self.dragged_item:
            source_values = self.tree.item(self.dragged_item)['values']
            target_values = self.tree.item(target)['values']
            
            if source_values and target_values:
                source_id = source_values[0]
                target_id = target_values[0]
                
                # Confirma vinculacao
                result = Messagebox.yesno(
                    title="Criar Vinculo Manual",
                    message=f"Vincular documento #{source_id} a #{target_id}?\n\n"
                            "Isso forcara a reconciliacao entre estes documentos.",
                    parent=self.root
                )
                

                if result == "Yes":
                    # Determine parent vs child based on type
                    # Parent: BOLETO or COMPROVANTE
                    # Child: NFE or NFSE
                    try:
                        source_id = int(source_id)
                        target_id = int(target_id)
                        
                        # Fetch types from DB for safety
                        cursor = self.db.connection.cursor()
                        cursor.execute("SELECT doc_type FROM documentos WHERE id IN (?, ?)", (source_id, target_id))
                        # We need to reuse the cursor correctly or just assume IDs are valid
                        # Better: use the helper
                        
                        # Helper specific for this check
                        def is_parent(did):
                             res = self.db.connection.execute("SELECT doc_type FROM documentos WHERE id=?", [did]).fetchone()
                             return res[0] in ['BOLETO', 'COMPROVANTE'] if res else False

                        if is_parent(source_id):
                            self.db.insert_match(source_id, target_id, MatchType.MANUAL, 1.0)
                        elif is_parent(target_id):
                            self.db.insert_match(target_id, source_id, MatchType.MANUAL, 1.0)
                        else:
                            # Both children or both parents? Arbitrary or fail
                            # Default to source=parent
                             self.db.insert_match(source_id, target_id, MatchType.MANUAL, 1.0)

                        # Auto-confirm manual matches
                        match_res = self.db.connection.execute("SELECT id FROM matches WHERE parent_doc_id=? AND child_doc_id=?", 
                                                              (source_id, target_id) if is_parent(source_id) else (target_id, source_id)).fetchone()
                        if match_res:
                            self.db.confirm_match(match_res[0])
                            
                    except Exception as e:
                        logger.error(f"Manual link error: {e}")
                        
                    self._refresh_all()
                    self._set_status(f"Vinculo manual criado: {source_id} <-> {target_id}")
        
        self.dragged_item = None
    
    # ==========================================================================
    # KEYBOARD SHORTCUTS (QoL Improvements)
    # ==========================================================================
    
    def _setup_keyboard_shortcuts(self):
        """Configure keyboard shortcuts for productivity (Vol I, Sec 7.2 - Human Efficiency)."""
        # Selection shortcuts
        self.root.bind("<Control-a>", lambda e: self._select_all())
        self.root.bind("<Control-A>", lambda e: self._select_all())  # Caps lock support
        self.root.bind("<Escape>", lambda e: self._clear_selection())
        
        # Action shortcuts
        self.root.bind("<Delete>", lambda e: self._on_delete_selected())
        self.root.bind("<F5>", lambda e: self._refresh_all())
        self.root.bind("<Control-e>", lambda e: self._on_edit_click())
        self.root.bind("<Control-E>", lambda e: self._on_edit_click())
        self.root.bind("<Control-o>", lambda e: self._on_open_file_click())
        self.root.bind("<Control-O>", lambda e: self._on_open_file_click())
        self.root.bind("<Control-r>", lambda e: self._on_bulk_reprocess())
        self.root.bind("<Control-R>", lambda e: self._on_bulk_reprocess())
        self.root.bind("<Control-p>", lambda e: self._on_process_selected_click())
        self.root.bind("<Control-P>", lambda e: self._on_process_selected_click())
        self.root.bind("<Control-b>", lambda e: self._on_bulk_edit_click())
        self.root.bind("<Control-B>", lambda e: self._on_bulk_edit_click())
        
        # Navigation
        self.root.bind("<Control-f>", lambda e: self._focus_filter())
        self.root.bind("<Control-F>", lambda e: self._focus_filter())
        
        # Additional QoL shortcuts
        self.root.bind("<Control-m>", lambda e: self._on_match_click())  # Match
        self.root.bind("<Control-M>", lambda e: self._on_match_click())
        self.root.bind("<Control-s>", lambda e: self._on_scan_click())   # Scan
        self.root.bind("<Control-S>", lambda e: self._on_scan_click())
        self.root.bind("<Control-g>", lambda e: self._on_show_graph_click())  # Graph
        self.root.bind("<Control-G>", lambda e: self._on_show_graph_click())
        self.root.bind("<F1>", lambda e: self._show_help())              # Help
        self.root.bind("<F2>", lambda e: self._on_edit_click())          # Quick Edit
        self.root.bind("<F3>", lambda e: self._quick_search_next())      # Find Next
        
        # Setup context menu for tree
        self._setup_context_menu()
        
        logger.info("Keyboard shortcuts configurados")
    
    def _setup_context_menu(self):
        """Setup right-click context menu for document tree."""
        self.context_menu = DocumentContextMenu(self)
        self.tree.bind('<Button-3>', self.context_menu.show)
    
    def _show_toast(self, message: str, style: str = 'info', duration: int = 3000):
        """Show a non-intrusive toast notification."""
        ToastNotification(self.root, message, style, duration)
    
    def _show_help(self):
        """Show keyboard shortcuts help dialog."""
        help_text = """
🎹 ATALHOS DE TECLADO

📂 ARQUIVOS
  Ctrl+O     Abrir arquivo selecionado
  Ctrl+S     Escanear pasta
  Ctrl+E     Editar documento
  F2         Edição rápida
  Delete     Excluir selecionado(s)

📋 SELEÇÃO
  Ctrl+A     Selecionar todos
  Escape     Limpar seleção

🔄 PROCESSAMENTO
  Ctrl+P     Processar selecionados
  Ctrl+R     Reprocessar
  Ctrl+M     Reconciliar (Match)
  Ctrl+B     Edição em lote

🔍 NAVEGAÇÃO
  Ctrl+F     Foco no filtro
  Ctrl+G     Mostrar grafo
  F1         Esta ajuda
  F3         Buscar próximo
  F5         Atualizar lista

🖱️ MOUSE
  Clique duplo     Editar
  Clique direito   Menu de contexto
  Arrastar         Reorganizar
"""
        Messagebox.showinfo(
            title="⌨️ Atalhos de Teclado",
            message=help_text,
            parent=self.root
        )
        return "break"
    
    def _quick_search_next(self):
        """Navigate to next search result or show search."""
        if hasattr(self, 'quick_search') and self.quick_search.get():
            # Find next matching item
            current = self.tree.focus()
            items = list(self.tree.get_children())
            if current in items:
                start = items.index(current) + 1
            else:
                start = 0
            
            query = self.quick_search.get().lower()
            for i in range(start, len(items)):
                values = self.tree.item(items[i])['values']
                if values and query in str(values).lower():
                    self.tree.selection_set(items[i])
                    self.tree.focus(items[i])
                    self.tree.see(items[i])
                    self._on_document_select(None)
                    return "break"
            
            self._show_toast("Nenhum resultado encontrado", 'warning')
        else:
            self._focus_filter()
        return "break"
    
    def _select_all(self):
        """Select all items in the TreeView (Ctrl+A)."""
        all_items = self.tree.get_children()
        if all_items:
            self.tree.selection_set(all_items)
            self._update_selection_status()
        return "break"  # Prevent default behavior
    
    def _clear_selection(self):
        """Clear all selections (Escape)."""
        self.tree.selection_remove(*self.tree.selection())
        self.selected_doc_id = None
        self._show_no_selection()
        self._set_status("Seleção limpa")
        return "break"
    
    def _focus_filter(self):
        """Focus on entity filter (Ctrl+F)."""
        self.filter_entity.focus_set()
        return "break"
    
    def _get_selected_doc_ids(self) -> List[int]:
        """Get list of all selected document IDs."""
        doc_ids = []
        for item in self.tree.selection():
            values = self.tree.item(item)['values']
            if values:
                doc_ids.append(values[0])  # ID is first column
        return doc_ids
    
    def _update_selection_status(self):
        """Update status bar to show selection count."""
        count = len(self.tree.selection())
        if count > 1:
            self._set_status(f"📋 {count} itens selecionados")
        elif count == 1:
            self._on_document_select(None)
    
    def _on_delete_selected(self):
        """Delete all selected documents (Delete key / bulk delete)."""
        doc_ids = self._get_selected_doc_ids()
        
        if not doc_ids:
            return
        
        count = len(doc_ids)
        
        if count == 1:
            # Single delete - use original method
            self.selected_doc_id = doc_ids[0]
            self._on_delete_click()
            return
        
        # Bulk delete
        result = Messagebox.yesno(
            title="Excluir Múltiplos",
            message=f"Excluir {count} documentos do banco?\n\n"
                    "⚠️ Arquivos originais NÃO serão apagados.\n"
                    "Esta ação não pode ser desfeita.",
            parent=self.root
        )
        
        if result == "Yes":
            deleted = 0
            for doc_id in doc_ids:
                try:
                    self._delete_document(doc_id)
                    deleted += 1
                except Exception as e:
                    logger.error(f"Erro ao excluir doc {doc_id}: {e}")
            
            self.selected_doc_id = None
            self._show_no_selection()
            self._refresh_all()
            self._set_status(f"✅ {deleted} documento(s) excluído(s)")
    
    def _on_bulk_reprocess(self):
        """Reprocess all selected documents (Ctrl+R)."""
        doc_ids = self._get_selected_doc_ids()
        
        if not doc_ids:
            return
        
        count = len(doc_ids)
        
        if count == 1:
            self.selected_doc_id = doc_ids[0]
            self._on_reprocess_click()
            return
        
        result = Messagebox.yesno(
            title="Reprocessar Múltiplos",
            message=f"Reprocessar {count} documentos?\n\n"
                    "Os dados atuais serão excluídos e os PDFs serão\n"
                    "reextraídos do zero.",
            parent=self.root
        )
        
        if result == "Yes":
            self._run_in_thread(self._do_bulk_reprocess, doc_ids)
    
    def _do_bulk_reprocess(self, doc_ids: List[int]):
        """Execute bulk reprocessing in background thread."""
        cursor = self.db.connection.cursor()
        processed = 0
        total = len(doc_ids)
        
        for i, doc_id in enumerate(doc_ids):
            self.root.after(0, lambda i=i, t=total: self._set_status(f"🔄 [{i+1}/{t}] Reprocessando..."))
            # Update TreeView status in real-time
            self.root.after(0, lambda did=doc_id: self._update_tree_item_status(did, "🔄 PROCESSANDO"))
            
            cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row and os.path.exists(row['original_path']):
                try:
                    self._delete_document(doc_id)
                    self.scanner.process_file(Path(row['original_path']))
                    processed += 1
                    self.root.after(0, lambda did=doc_id: self._update_tree_item_status(did, "✅ PARSED"))
                except Exception as e:
                    logger.error(f"Erro ao reprocessar doc {doc_id}: {e}")
                    self.root.after(0, lambda did=doc_id: self._update_tree_item_status(did, "❌ ERRO"))
        
        self.root.after(0, lambda: self._set_status(f"✅ {processed}/{total} reprocessado(s)"))
        self.root.after(0, self._refresh_document_list)
    
    def _update_tree_item_status(self, doc_id: int, status: str):
        """Atualiza o status de um item no TreeView em tempo real."""
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            if values and values[0] == doc_id:
                # Update only the status column (index 5)
                new_values = list(values)
                new_values[5] = status
                self.tree.item(item, values=new_values)
                self.tree.see(item)  # Scroll to visible
                break
    
    def _on_process_selected_click(self):
        """Process selected documents with AI extraction (OCR + Ensemble)."""
        selection = self.tree.selection()
        if not selection:
            # Process all pending documents
            cursor = self.db.connection.cursor()
            cursor.execute("SELECT id FROM documentos WHERE status = 'PENDING'")
            rows = cursor.fetchall()
            if not rows:
                Messagebox.showinfo("Info", "Nenhum documento pendente para processar.", parent=self.root)
                return
            doc_ids = [row[0] for row in rows]
            if len(doc_ids) > 20:
                result = Messagebox.yesno(
                    "Processar Todos", 
                    f"Processar {len(doc_ids)} documentos pendentes com IA?\n\nIsso pode demorar.",
                    parent=self.root
                )
                if result != "Yes":
                    return
        else:
            doc_ids = []
            for item in selection:
                values = self.tree.item(item)['values']
                if values:
                    doc_ids.append(values[0])
        
        if not doc_ids:
            return
        
        if self.is_processing:
            return
        
        self._run_in_thread(self._do_process_documents, doc_ids)
    
    def _do_process_documents(self, doc_ids: List[int]):
        """Process documents with AI extraction in background thread."""
        import gc
        
        cursor = self.db.connection.cursor()
        total = len(doc_ids)
        processed = 0
        errors = 0
        
        self.gui_queue.put(("status", f"🧠 Iniciando processamento de {total} documento(s)..."))
        
        for i, doc_id in enumerate(doc_ids):
            # Update status
            self.gui_queue.put(("status", f"🧠 [{i+1}/{total}] Processando documento..."))
            self.gui_queue.put(("tree_update", (doc_id, "🔄 AI...")))
            
            try:
                # Get document info
                cursor.execute("""
                    SELECT original_path, page_start, page_end, doc_type
                    FROM documentos WHERE id = ?
                """, (doc_id,))
                row = cursor.fetchone()
                
                if not row:
                    continue
                
                file_path = row[0] if isinstance(row, tuple) else row['original_path']
                page_start = row[1] if isinstance(row, tuple) else row['page_start']
                doc_type_str = row[3] if isinstance(row, tuple) else row['doc_type']
                
                if not file_path or not os.path.exists(file_path):
                    errors += 1
                    self.gui_queue.put(("tree_update", (doc_id, "❌ NoFile")))
                    continue
                
                # Use hierarchical_extract from scanner (new method)
                from pathlib import Path
                try:
                    doc_type = DocumentType[doc_type_str] if doc_type_str else DocumentType.UNKNOWN
                except:
                    doc_type = DocumentType.UNKNOWN
                
                result = self.scanner.hierarchical_extract(
                    Path(file_path),
                    page_num=(page_start or 1) - 1,
                    doc_type=doc_type
                )
                
                # Update database with extracted data
                if result['amount_cents'] > 0 or result['due_date'] or result['cnpj']:
                    # Update or insert transaction
                    cursor.execute("""
                        SELECT id FROM transacoes WHERE doc_id = ?
                    """, (doc_id,))
                    trans_row = cursor.fetchone()
                    
                    if trans_row:
                        # Update existing transaction
                        cursor.execute("""
                            UPDATE transacoes SET
                                amount_cents = COALESCE(NULLIF(?, 0), amount_cents),
                                due_date = COALESCE(?, due_date),
                                emission_date = COALESCE(?, emission_date)
                            WHERE doc_id = ?
                        """, (result['amount_cents'], result['due_date'], result['emission_date'], doc_id))
                    else:
                        # Insert new transaction
                        cursor.execute("""
                            INSERT INTO transacoes (doc_id, amount_cents, due_date, emission_date)
                            VALUES (?, ?, ?, ?)
                        """, (doc_id, result['amount_cents'], result['due_date'], result['emission_date']))
                    
                    # Update entity if found
                    if result.get('entity_tag'):
                        entity_val = result['entity_tag'].value if hasattr(result['entity_tag'], 'value') else str(result['entity_tag'])
                        cursor.execute("""
                            UPDATE documentos SET entity_tag = ? WHERE id = ?
                        """, (entity_val, doc_id))
                    
                # Update document status
                new_status = 'PARSED'
                status_msg = "✅ PARSED"
                
                if result.get('review_needed'):
                    status_msg = "⚠️ REVIEW"
                    # Opcional: Poderiamos ter um status 'REVIEW' no DB, mas por enquanto mantemos PARSED
                    # porem logamos ou marcamos de alguma forma.
                
                if result['amount_cents'] > 0 or result['due_date'] or result['cnpj']:
                    # Update or insert transaction
                    cursor.execute("""
                        SELECT id FROM transacoes WHERE doc_id = ?
                    """, (doc_id,))
                    trans_row = cursor.fetchone()
                    
                    if trans_row:
                        # Update existing transaction
                        cursor.execute("""
                            UPDATE transacoes SET
                                amount_cents = COALESCE(NULLIF(?, 0), amount_cents),
                                due_date = COALESCE(?, due_date),
                                emission_date = COALESCE(?, emission_date)
                            WHERE doc_id = ?
                        """, (result['amount_cents'], result['due_date'], result['emission_date'], doc_id))
                    else:
                        # Insert new transaction
                        cursor.execute("""
                            INSERT INTO transacoes (doc_id, amount_cents, due_date, emission_date)
                            VALUES (?, ?, ?, ?)
                        """, (doc_id, result['amount_cents'], result['due_date'], result['emission_date']))
                    
                    # Update entity if found
                    if result.get('entity_tag'):
                        entity_val = result['entity_tag'].value if hasattr(result['entity_tag'], 'value') else str(result['entity_tag'])
                        cursor.execute("""
                            UPDATE documentos SET entity_tag = ? WHERE id = ?
                        """, (entity_val, doc_id))
                    
                    # Update document status
                    cursor.execute("""
                        UPDATE documentos SET status = 'PARSED' WHERE id = ?
                    """, (doc_id,))
                    
                    self.db.connection.commit()
                    
                    # Update tree with extracted data
                    self.gui_queue.put(("tree_update", (doc_id, status_msg)))
                    processed += 1
                else:
                    # No data extracted
                    cursor.execute("""
                        UPDATE documentos SET status = 'PARSED' WHERE id = ?
                    """, (doc_id,))
                    self.db.connection.commit()
                    self.gui_queue.put(("tree_update", (doc_id, "⚠️ EMPTY")))
                    processed += 1
                
                # Memory cleanup after each document
                gc.collect()
                
            except Exception as e:
                logger.error(f"Erro ao processar doc {doc_id}: {e}")
                errors += 1
                self.gui_queue.put(("tree_update", (doc_id, "❌ ERRO")))
        
        # Final status
        msg = f"✅ Processado: {processed}/{total} | Erros: {errors}"
        self.gui_queue.put(("status", msg))
        self.gui_queue.put(("refresh_all", None))
    
    def _update_tree_item_full(self, doc_id: int, tipo: str = None, entity: str = None, 
                                fornecedor: str = None, valor: str = None, status: str = None):
        """Atualiza TODAS as colunas de um item no TreeView em tempo real.
        
        Columns: [id, tipo, ent, fornecedor, valor, status]
        """
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            if values and values[0] == doc_id:
                new_values = list(values)
                # Update each column if provided (indices: 0=id, 1=tipo, 2=ent, 3=fornecedor, 4=valor, 5=status)
                if tipo is not None:
                    new_values[1] = tipo
                if entity is not None:
                    new_values[2] = entity
                if fornecedor is not None:
                    new_values[3] = fornecedor[:25] if len(fornecedor) > 25 else fornecedor
                if valor is not None:
                    new_values[4] = valor
                if status is not None:
                    new_values[5] = status
                
                self.tree.item(item, values=new_values)
                self.tree.see(item)  # Scroll to visible
                self.tree.update_idletasks()  # Force visual update
                break
    
    def _on_process_selected_click(self):
        """Processa selecionados com AI (OCR + GLiNER)."""
        if self.is_processing:
            return
        
        doc_ids = self._get_selected_doc_ids()
        
        if not doc_ids:
            # Se nada selecionado, processa todos pendentes
            cursor = self.db.connection.cursor()
            cursor.execute("SELECT id FROM documentos WHERE status = 'PENDING' OR status = 'IMPORTED'")
            doc_ids = [row['id'] for row in cursor.fetchall()]
            
            if not doc_ids:
                self._set_status("⚠️ Nenhum documento pendente para processar")
                return
            
            result = Messagebox.yesno(
                title="Processar Pendentes",
                message=f"Processar {len(doc_ids)} documentos pendentes com AI?\n\n"
                        "Isso pode levar alguns minutos.",
                parent=self.root
            )
            if result != "Yes":
                return
        
        self._run_in_thread(self._do_process_selected, doc_ids)
    
    def _do_process_selected(self, doc_ids: List[int]):
        """Processa documentos selecionados com AI e atualiza TreeView em tempo real."""
        if not self.scanner.ensemble:
            self.root.after(0, lambda: self._set_status("⚠️ EnsembleExtractor não disponível"))
            return
        
        total = len(doc_ids)
        processed = 0
        cursor = self.db.connection.cursor()
        
        for i, doc_id in enumerate(doc_ids):
            # Busca info do documento
            cursor.execute("SELECT original_path, page_start, page_end, raw_text FROM documentos WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if not row:
                continue
            
            filename = Path(row['original_path']).name if row['original_path'] else f"Doc #{doc_id}"
            
            # Update status: PROCESSING
            self.root.after(0, lambda i=i, t=total, fn=filename: self._set_status(f"🧠 [{i+1}/{t}] Analisando: {fn[:35]}..."))
            self.root.after(0, lambda did=doc_id: self._update_tree_item_status(did, "🧠 ANALISANDO"))
            
            try:
                # Extrai com EnsembleExtractor (OCR + GLiNER)
                ensemble_result = self.scanner.ensemble.extract_from_pdf(
                    row['original_path'],
                    page_range=(row['page_start'], row['page_end']) if row['page_start'] else None
                )
                
                if ensemble_result:
                    # Format value for display
                    valor_display = ""
                    if ensemble_result.amount_cents > 0:
                        valor_display = f"R$ {ensemble_result.amount_cents / 100:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    
                    # Get doc type and entity (handle both string and Enum types)
                    doc_type_raw = ensemble_result.doc_type
                    entity_raw = ensemble_result.entity_tag
                    
                    tipo_display = getattr(doc_type_raw, 'value', doc_type_raw) if doc_type_raw else "UNKNOWN"
                    entity_display = getattr(entity_raw, 'value', entity_raw) if entity_raw else ""
                    fornecedor_display = ensemble_result.fornecedor or "?"
                    
                    # UPDATE TREEVIEW EM TEMPO REAL - Cascading visual feedback!
                    self.root.after(0, lambda did=doc_id, t=tipo_display, e=entity_display, f=fornecedor_display, v=valor_display: 
                        self._update_tree_item_full(did, tipo=t, entity=e, fornecedor=f, valor=v, status="✅ PARSED"))
                    
                    # Update database
                    if ensemble_result.amount_cents > 0:
                        cursor.execute("""
                            UPDATE transacoes SET 
                                supplier_clean = ?,
                                amount_cents = COALESCE(?, amount_cents),
                                due_date = COALESCE(?, due_date),
                                emission_date = COALESCE(?, emission_date)
                            WHERE doc_id = ?
                        """, (
                            ensemble_result.fornecedor,
                            ensemble_result.amount_cents if ensemble_result.amount_cents > 0 else None,
                            ensemble_result.due_date,
                            ensemble_result.emission_date,
                            doc_id
                        ))
                    
                    # Update document type, status, and doc_number
                    cursor.execute("""
                        UPDATE documentos SET 
                            status = 'PARSED',
                            doc_type = COALESCE(?, doc_type),
                            entity_tag = COALESCE(?, entity_tag),
                            doc_number = COALESCE(?, doc_number)
                        WHERE id = ?
                    """, (tipo_display, entity_display or None, ensemble_result.doc_number, doc_id))
                    self.db.connection.commit()
                    
                    processed += 1
                    
                    # Log para debug
                    print(f"  ✅ [{doc_id}] {tipo_display} | {entity_display} | {fornecedor_display} | {valor_display}")
                else:
                    self.root.after(0, lambda did=doc_id: self._update_tree_item_full(did, status="⚠️ PARCIAL"))
                    processed += 1
                    
            except Exception as e:
                logger.error(f"Erro AI doc {doc_id}: {e}")
                self.root.after(0, lambda did=doc_id: self._update_tree_item_full(did, status="❌ ERRO"))
        
        self.root.after(0, lambda: self._set_status(f"✅ Processado: {processed}/{total} documentos com AI"))
        self.root.after(0, self._refresh_document_list)
    
    def _on_bulk_edit_click(self):
        """Edição em massa de múltiplos documentos selecionados."""
        doc_ids = self._get_selected_doc_ids()
        
        if len(doc_ids) < 2:
            Messagebox.showinfo("Info", "Selecione 2 ou mais documentos para edição em massa", parent=self.root)
            return
        
        # Cria janela de dialogo
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edição em Massa - {len(doc_ids)} documentos")
        dialog.geometry("450x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Frame principal
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)
        
        ttk.Label(main_frame, text=f"📝 Editando {len(doc_ids)} documentos", font=("Helvetica", 14, "bold")).pack(pady=(0, 15))
        ttk.Label(main_frame, text="Deixe em branco para manter valor atual", bootstyle="secondary").pack(pady=(0, 15))
        
        # Fornecedor
        ttk.Label(main_frame, text="Fornecedor:").pack(anchor=W)
        supplier_var = tk.StringVar()
        supplier_entry = ttk.Entry(main_frame, textvariable=supplier_var, width=50)
        supplier_entry.pack(fill=X, pady=(0, 10))
        
        # Entidade
        ttk.Label(main_frame, text="Entidade:").pack(anchor=W)
        entity_var = tk.StringVar(value="(manter)")
        entity_combo = ttk.Combobox(main_frame, textvariable=entity_var, values=["(manter)", "VG", "MV"], state="readonly", width=15)
        entity_combo.pack(anchor=W, pady=(0, 10))
        
        # Data
        ttk.Label(main_frame, text="Data (DD/MM/AAAA):").pack(anchor=W)
        date_var = tk.StringVar()
        date_entry = ttk.Entry(main_frame, textvariable=date_var, width=20)
        date_entry.pack(anchor=W, pady=(0, 15))
        
        # Aplicar
        def apply_bulk_edit():
            supplier = supplier_var.get().strip()
            entity = entity_var.get() if entity_var.get() != "(manter)" else None
            date_str = date_var.get().strip()
            
            # Converte data para ISO
            iso_date = None
            if date_str:
                parts = date_str.replace(".", "/").split("/")
                if len(parts) == 3:
                    iso_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
            
            cursor = self.db.connection.cursor()
            updated = 0
            
            for doc_id in doc_ids:
                try:
                    # Atualiza transacao
                    updates = []
                    params = []
                    
                    if supplier:
                        updates.append("supplier_clean = ?")
                        params.append(supplier.upper())
                    
                    if iso_date:
                        updates.append("due_date = ?")
                        params.append(iso_date)
                    
                    if updates:
                        params.append(doc_id)
                        cursor.execute(f"UPDATE transacoes SET {', '.join(updates)} WHERE doc_id = ?", tuple(params))
                    
                    # Atualiza entidade no documento
                    if entity:
                        cursor.execute("UPDATE documentos SET entity_tag = ? WHERE id = ?", (entity, doc_id))
                    
                    updated += 1
                except Exception as e:
                    logger.error(f"Erro atualizando doc {doc_id}: {e}")
            
            self.db.connection.commit()
            dialog.destroy()
            self._refresh_all()
            self._set_status(f"✅ {updated} documentos atualizados em massa!")
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Cancelar", bootstyle="secondary", command=dialog.destroy).pack(side=RIGHT, padx=5)
        ttk.Button(btn_frame, text="✅ Aplicar a Todos", bootstyle="success", command=apply_bulk_edit).pack(side=RIGHT, padx=5)
    
    # ==========================================================================
    # MENU DE CONTEXTO
    # ==========================================================================
    
    def _create_context_menu(self):
        self.context_menu = tk.Menu(self.root, tearoff=0)
        # Single-item commands (indices 0-8)
        self.context_menu.add_command(label="📝 Editar Dados...", command=self._on_edit_click)
        self.context_menu.add_command(label="✏️ Editar em Massa...", command=self._on_bulk_edit_click)  # NEW
        self.context_menu.add_command(label="🧠 Processar com AI", command=self._on_process_selected_click)  # NEW
        self.context_menu.add_command(label="🔄 Reprocessar PDF", command=self._on_reprocess_click)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="📂 Abrir Arquivo", command=self._on_open_file_click)
        self.context_menu.add_command(label="📁 Abrir Pasta", command=self._on_open_folder_click)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="🔗 Ver Vinculos no Grafo", command=self._on_show_graph_click)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="🗑️ Excluir", command=self._on_delete_selected)
        
        self.tree.bind("<Button-3>", self._show_context_menu)
    
    def _show_context_menu(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        
        # If clicking on unselected item, select only that item
        if item not in self.tree.selection():
            self.tree.selection_set(item)
        
        # Get selection count and update menu labels dynamically
        count = len(self.tree.selection())
        
        if count > 1:
            # Bulk mode - update labels (indices: 0=Edit, 1=BulkEdit, 2=ProcessAI, 3=Reprocess, 4=sep, 5=Open, 6=OpenFolder, 7=sep, 8=Graph, 9=sep, 10=Delete)
            self.context_menu.entryconfig(0, label=f"📝 Editar (selecione 1)", state=DISABLED)
            self.context_menu.entryconfig(1, label=f"✏️ Editar {count} em Massa...", state=NORMAL)
            self.context_menu.entryconfig(2, label=f"🧠 Processar {count} com AI", state=NORMAL)
            self.context_menu.entryconfig(3, label=f"🔄 Reprocessar {count} PDFs", command=self._on_bulk_reprocess)
            self.context_menu.entryconfig(5, label=f"📂 Abrir (selecione 1)", state=DISABLED)
            self.context_menu.entryconfig(6, label=f"📁 Abrir Pasta (selecione 1)", state=DISABLED)
            self.context_menu.entryconfig(8, label=f"🔗 Ver Grafo (selecione 1)", state=DISABLED)
            self.context_menu.entryconfig(10, label=f"🗑️ Excluir {count} itens")
        else:
            # Single mode - restore labels
            self.context_menu.entryconfig(0, label="📝 Editar Dados...", state=NORMAL, command=self._on_edit_click)
            self.context_menu.entryconfig(1, label="✏️ Editar em Massa...", state=NORMAL)
            self.context_menu.entryconfig(2, label="🧠 Processar com AI", state=NORMAL)
            self.context_menu.entryconfig(3, label="🔄 Reprocessar PDF", command=self._on_reprocess_click, state=NORMAL)
            self.context_menu.entryconfig(5, label="📂 Abrir Arquivo", state=NORMAL)
            self.context_menu.entryconfig(6, label="📁 Abrir Pasta", state=NORMAL)
            self.context_menu.entryconfig(8, label="🔗 Ver Vinculos no Grafo", state=NORMAL)
            self.context_menu.entryconfig(10, label="🗑️ Excluir")
            self._on_document_select(None)
        
        self.context_menu.post(event.x_root, event.y_root)
    
    # ==========================================================================
    # CONSTRUCAO DA INTERFACE
    # ==========================================================================
    
    def _build_ui(self):
        self.main_container = ttk.Frame(self.root, padding=10)
        self.main_container.pack(fill=BOTH, expand=YES)
        
        self._build_header()
        ttk.Separator(self.main_container, orient=HORIZONTAL).pack(fill=X, pady=10)
        self._build_content_area()
        self._build_status_bar()
    
    def _build_header(self):
        """Build premium header with stats dashboard."""
        header = ttk.Frame(self.main_container)
        header.pack(fill=X, pady=(0, 5))
        
        # =============== LEFT: Logo & Title ===============
        brand_frame = ttk.Frame(header)
        brand_frame.pack(side=LEFT)
        
        # Main title with gradient effect
        title_label = ttk.Label(
            brand_frame, 
            text="🏛️ SDRA-Rural", 
            font=("Segoe UI", 22, "bold"), 
            bootstyle="inverse-primary"
        )
        title_label.pack(side=LEFT, padx=(0, 8))
        
        # Version badge
        version_frame = ttk.Frame(brand_frame)
        version_frame.pack(side=LEFT, padx=(0, 15))
        
        ttk.Label(
            version_frame, 
            text="v3.0", 
            font=("Consolas", 9, "bold"),
            bootstyle="success"
        ).pack()
        
        ttk.Label(
            version_frame, 
            text="Premium Edition", 
            font=("Segoe UI", 8),
            bootstyle="secondary"
        ).pack()
        
        # Status indicators
        status_frame = ttk.Frame(brand_frame)
        status_frame.pack(side=LEFT, padx=(0, 10))
        
        dnd_icon = "✅" if DND_AVAILABLE else "❌"
        dnd_text = "Drag & Drop" if DND_AVAILABLE else "DnD OFF"
        ttk.Label(
            status_frame, 
            text=f"{dnd_icon} {dnd_text}", 
            font=("Segoe UI", 9),
            bootstyle="success" if DND_AVAILABLE else "danger"
        ).pack()
        
        gmail_icon = "📧" if GMAIL_AVAILABLE else "📭"
        gmail_text = "Gmail Ready" if GMAIL_AVAILABLE else "Gmail OFF"
        ttk.Label(
            status_frame, 
            text=f"{gmail_icon} {gmail_text}", 
            font=("Segoe UI", 9),
            bootstyle="info" if GMAIL_AVAILABLE else "secondary"
        ).pack()
        
        # =============== CENTER: Stats Dashboard ===============
        stats_frame = ttk.Frame(header)
        stats_frame.pack(side=LEFT, expand=YES, padx=20)
        
        # Stats cards
        self.stat_total = self._create_stat_card(stats_frame, "📄", "0", "Total")
        self.stat_pending = self._create_stat_card(stats_frame, "⏳", "0", "Pendentes")
        self.stat_reconciled = self._create_stat_card(stats_frame, "🔗", "0", "Reconciliados")
        self.stat_complete = self._create_stat_card(stats_frame, "✅", "0", "Completos")
        
        # =============== RIGHT: Action Toolbar ===============
        toolbar = ttk.Frame(header)
        toolbar.pack(side=RIGHT)
        
        # Primary actions group
        primary_group = ttk.Frame(toolbar)
        primary_group.pack(side=LEFT, padx=(0, 10))
        
        self.btn_add = ttk.Button(
            primary_group, 
            text="📥 Importar", 
            bootstyle="primary", 
            width=11, 
            command=self._on_add_files_click
        )
        self.btn_add.pack(side=LEFT, padx=2)
        ToolTip(self.btn_add, text="Adiciona PDFs (ou arraste para janela)")
        
        self.btn_scan = ttk.Button(
            primary_group, 
            text="📁 Pasta", 
            bootstyle="primary-outline", 
            width=9, 
            command=self._on_scan_click
        )
        self.btn_scan.pack(side=LEFT, padx=2)
        ToolTip(self.btn_scan, text="Escaneia pasta inteira (Ctrl+S)")
        
        # Processing actions group
        process_group = ttk.Frame(toolbar)
        process_group.pack(side=LEFT, padx=(0, 10))
        
        ttk.Separator(toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=5)
        
        self.btn_process = ttk.Button(
            process_group, 
            text="🧠 AI", 
            bootstyle="warning", 
            width=7, 
            command=self._on_process_selected_click
        )
        self.btn_process.pack(side=LEFT, padx=2)
        ToolTip(self.btn_process, text="Processa com OCR + Gemini (Ctrl+P)")
        
        self.btn_match = ttk.Button(
            process_group, 
            text="🔗 Match", 
            bootstyle="success", 
            width=9, 
            command=self._on_match_click
        )
        self.btn_match.pack(side=LEFT, padx=2)
        ToolTip(self.btn_match, text="Reconcilia via MCMF (Ctrl+M)")
        
        self.btn_rename = ttk.Button(
            process_group, 
            text="📋 Renomear", 
            bootstyle="info-outline", 
            width=11, 
            command=self._on_rename_click
        )
        self.btn_rename.pack(side=LEFT, padx=2)
        ToolTip(self.btn_rename, text="Renomeia arquivos reconciliados")
        
        # Gmail integration
        ttk.Separator(toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=5)
        
        gmail_group = ttk.Frame(toolbar)
        gmail_group.pack(side=LEFT, padx=(0, 10))
        
        self.btn_gmail_sync = ttk.Button(
            gmail_group, 
            text="📧 Gmail", 
            bootstyle="info", 
            width=9, 
            command=self._on_gmail_sync_click,
            state=NORMAL if GMAIL_AVAILABLE else DISABLED
        )
        self.btn_gmail_sync.pack(side=LEFT, padx=2)
        ToolTip(self.btn_gmail_sync, text="Sincroniza emails do Gmail (Project Cyborg)")
        
        # Quick actions
        quick_group = ttk.Frame(toolbar)
        quick_group.pack(side=LEFT)
        
        self.btn_refresh = ttk.Button(
            quick_group, 
            text="↻", 
            bootstyle="secondary-outline", 
            width=3, 
            command=self._refresh_all
        )
        self.btn_refresh.pack(side=LEFT, padx=2)
        ToolTip(self.btn_refresh, text="Atualizar lista (F5)")
        
        self.btn_help = ttk.Button(
            quick_group, 
            text="?", 
            bootstyle="secondary-outline", 
            width=3, 
            command=self._show_help
        )
        self.btn_help.pack(side=LEFT, padx=2)
        ToolTip(self.btn_help, text="Ajuda e atalhos (F1)")
    
    def _create_stat_card(self, parent, icon: str, value: str, label: str) -> ttk.Label:
        """Create a mini stat card for the dashboard."""
        card = ttk.Frame(parent)
        card.pack(side=LEFT, padx=8)
        
        # Icon + Value
        top = ttk.Frame(card)
        top.pack()
        
        ttk.Label(top, text=icon, font=("Segoe UI Emoji", 12)).pack(side=LEFT)
        value_label = ttk.Label(top, text=value, font=("Consolas", 14, "bold"), bootstyle="primary")
        value_label.pack(side=LEFT, padx=(5, 0))
        
        # Label
        ttk.Label(card, text=label, font=("Segoe UI", 8), bootstyle="secondary").pack()
        
        return value_label
    
    def _update_stats_dashboard(self):
        """Update stats dashboard with current counts."""
        try:
            cursor = self.db.connection.cursor()
            
            # Total documents
            cursor.execute("SELECT COUNT(*) FROM documentos WHERE status != 'ARCHIVED'")
            total = cursor.fetchone()[0]
            
            # Pending (not reconciled)
            cursor.execute("SELECT COUNT(*) FROM documentos WHERE status IN ('INGESTED', 'PARSED')")
            pending = cursor.fetchone()[0]
            
            # Reconciled
            cursor.execute("SELECT COUNT(*) FROM documentos WHERE status = 'RECONCILED'")
            reconciled = cursor.fetchone()[0]
            
            # Complete (renamed)
            cursor.execute("SELECT COUNT(*) FROM documentos WHERE status = 'RENAMED'")
            complete = cursor.fetchone()[0]
            
            # Update labels
            if hasattr(self, 'stat_total'):
                self.stat_total.config(text=str(total))
                self.stat_pending.config(text=str(pending))
                self.stat_reconciled.config(text=str(reconciled))
                self.stat_complete.config(text=str(complete))
        except:
            pass
    
    def _build_content_area(self):
        content = ttk.Frame(self.main_container)
        content.pack(fill=BOTH, expand=YES)
        
        content.columnconfigure(0, weight=50)
        content.columnconfigure(1, weight=50)
        content.rowconfigure(0, weight=1)
        
        self._build_document_list(content)
        self._build_details_panel(content)
    
    def _build_document_list(self, parent):
        left_frame = ttk.Labelframe(parent, text=" Documentos (Arraste para Vincular) ", padding=10, bootstyle="primary")
        left_frame.grid(row=0, column=0, sticky=NSEW, padx=(0, 5))
        
        # Filtros
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(filter_frame, text="Entidade:").pack(side=LEFT, padx=(0, 5))
        self.filter_entity = ttk.Combobox(filter_frame, values=["Todos", "VG", "MV"], state="readonly", width=10)
        self.filter_entity.set("Todos")
        self.filter_entity.pack(side=LEFT, padx=(0, 15))
        self.filter_entity.bind("<<ComboboxSelected>>", lambda e: self._refresh_document_list())
        
        ttk.Label(filter_frame, text="Tipo:").pack(side=LEFT, padx=(0, 5))
        self.filter_type = ttk.Combobox(filter_frame, values=["Todos", "NFE", "NFSE", "BOLETO", "COMPROVANTE"], state="readonly", width=12)
        self.filter_type.set("Todos")
        self.filter_type.pack(side=LEFT, padx=(0, 15))
        self.filter_type.bind("<<ComboboxSelected>>", lambda e: self._refresh_document_list())
        
        ttk.Label(filter_frame, text="Status:").pack(side=LEFT, padx=(0, 5))
        self.filter_status = ttk.Combobox(filter_frame, values=["Todos", "PARSED", "RECONCILED", "RENAMED"], state="readonly", width=12)
        self.filter_status.set("Todos")
        self.filter_status.pack(side=LEFT)
        self.filter_status.bind("<<ComboboxSelected>>", lambda e: self._refresh_document_list())
        
        # TreeView
        columns = ("id", "tipo", "ent", "fornecedor", "valor", "status")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", bootstyle="primary", selectmode="extended")  # QoL: Multi-select enabled
        
        self.tree.heading("id", text="ID")
        self.tree.heading("tipo", text="Tipo")
        self.tree.heading("ent", text="Ent")
        self.tree.heading("fornecedor", text="Fornecedor")
        self.tree.heading("valor", text="Valor")
        self.tree.heading("status", text="Status")
        
        self.tree.column("id", width=40, anchor=CENTER)
        self.tree.column("tipo", width=70, anchor=CENTER)
        self.tree.column("ent", width=35, anchor=CENTER)
        self.tree.column("fornecedor", width=180)
        self.tree.column("valor", width=100, anchor=E)
        self.tree.column("status", width=90, anchor=CENTER)
        
        scrollbar = ttk.Scrollbar(left_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_selection_change)
        self.tree.bind("<Double-1>", lambda e: self._on_edit_click())                                                               
    
    def _build_details_panel(self, parent):
        right_frame = ttk.Labelframe(parent, text=" Detalhes & Vinculos ", padding=10, bootstyle="info")
        right_frame.grid(row=0, column=1, sticky=NSEW, padx=(5, 0))
        
        # Notebook
        self.notebook = ttk.Notebook(right_frame, bootstyle="info")
        self.notebook.pack(fill=BOTH, expand=YES)
        
        # Aba: Detalhes
        self.details_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.details_tab, text=" 📋 Detalhes ")
        
        self.details_scroll = ScrolledFrame(self.details_tab, autohide=True)
        self.details_scroll.pack(fill=BOTH, expand=YES)
        self.details_frame = self.details_scroll
        
        # Aba: Preview
        self.preview_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.preview_tab, text=" 📄 Preview ")
        
        self.preview_label = ttk.Label(self.preview_tab, anchor=CENTER)
        self.preview_label.pack(fill=BOTH, expand=YES)
        
        # Aba: Grafo/Vinculos
        self.graph_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.graph_tab, text=" 🔗 Ilha de Transacao ")
        
        self.graph_text = tk.Text(self.graph_tab, wrap=WORD, font=("Consolas", 10))
        self.graph_text.pack(fill=BOTH, expand=YES)
        
        # Aba: Email Monitor (Project Cyborg)
        self.email_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.email_tab, text=" 📧 Email Monitor ")
        self._build_email_tab()
        
        # Aba: Conciliação Visual (Sprint 4)
        self.conciliation_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.conciliation_tab, text=" 🔄 Conciliação ")
        self._build_conciliation_tab()
        
        # Botoes de acao
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=X, pady=(10, 0))
        
        ttk.Button(action_frame, text="Editar", bootstyle="info-outline", command=self._on_edit_click, width=10).pack(side=LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Abrir PDF", bootstyle="secondary-outline", command=self._on_open_file_click, width=10).pack(side=LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Excluir", bootstyle="danger-outline", command=self._on_delete_click, width=10).pack(side=RIGHT)
        
        self._show_no_selection()
    
    def _build_email_tab(self):
        """Build the Email Monitor tab for Project Cyborg."""
        # Control buttons frame
        control_frame = ttk.Frame(self.email_tab)
        control_frame.pack(fill=X, padx=5, pady=5)
        
        self.btn_start_email = ttk.Button(
            control_frame,
            text="▶ Iniciar Monitoramento",
            command=self._on_start_email_monitor,
            bootstyle="success"
        )
        self.btn_start_email.pack(side=LEFT, padx=5)
        
        self.btn_stop_email = ttk.Button(
            control_frame,
            text="⏹ Parar",
            command=self._on_stop_email_monitor,
            bootstyle="danger",
            state=DISABLED
        )
        self.btn_stop_email.pack(side=LEFT, padx=5)
        
        # Sync Now button (Sprint 6)
        self.btn_sync_now = ttk.Button(
            control_frame,
            text="🔄 Sincronizar Agora",
            command=self._on_gmail_sync_click,
            bootstyle="info"
        )
        self.btn_sync_now.pack(side=LEFT, padx=5)
        
        # Token counter
        self.lbl_gemini_tokens = ttk.Label(
            control_frame,
            text="Tokens Gemini: 0",
            bootstyle="info"
        )
        self.lbl_gemini_tokens.pack(side=RIGHT, padx=10)
        
        # Stats labels
        self.lbl_email_stats = ttk.Label(
            control_frame,
            text="Emails: 0 | Docs: 0",
            bootstyle="secondary"
        )
        self.lbl_email_stats.pack(side=RIGHT, padx=10)
        
        # Main content area (split between triage and log)
        content_pane = ttk_native.PanedWindow(self.email_tab, orient=VERTICAL)
        content_pane.pack(fill=BOTH, expand=YES, padx=5, pady=5)
        
        # === TOP: Email Triage Area (Sprint 6) ===
        triage_frame = ttk.Labelframe(content_pane, text=" Triagem de Emails ", padding=5)
        content_pane.add(triage_frame, weight=60)
        
        # Scrollable area for email cards
        self.email_cards_scroll = ScrolledFrame(triage_frame, autohide=True)
        self.email_cards_scroll.pack(fill=BOTH, expand=YES)
        self.email_cards_container = self.email_cards_scroll
        
        # Empty state message
        self.email_empty_label = ttk.Label(
            self.email_cards_container,
            text="📭 Nenhum email na triagem.\n\nClique em 'Sincronizar Agora' para buscar emails do Gmail.",
            font=('Helvetica', 11),
            anchor=CENTER,
            justify=CENTER,
            bootstyle='secondary'
        )
        self.email_empty_label.pack(expand=YES, pady=50)
        
        # === BOTTOM: Log Console ===
        log_frame = ttk.Labelframe(content_pane, text=" Console de Log ", padding=5)
        content_pane.add(log_frame, weight=40)
        
        # Text widget with scrollbar
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=RIGHT, fill=Y)
        
        self.email_log = tk.Text(
            log_frame,
            wrap=WORD,
            height=8,
            bg='#1e1e1e',
            fg='#ffffff',
            font=('Consolas', 10),
            yscrollcommand=log_scroll.set
        )
        self.email_log.pack(fill=BOTH, expand=YES)
        log_scroll.config(command=self.email_log.yview)
        
        # Configure log colors
        self.email_log.tag_config('info', foreground='#3498db')
        self.email_log.tag_config('success', foreground='#2ecc71')
        self.email_log.tag_config('warning', foreground='#f39c12')
        self.email_log.tag_config('error', foreground='#e74c3c')
        self.email_log.tag_config('gemini', foreground='#9b59b6')
        self.email_log.tag_config('timestamp', foreground='#7f8c8d')
        
        # Initial message
        self._email_log_message("Pipeline de email pronto. Clique 'Sincronizar Agora' para começar.", 'info')
        
        # Pipeline reference
        self._email_pipeline = None
        self._email_thread = None
        self._gemini_total_tokens = 0
        self._email_cards = []  # Track email cards
    
    def _add_email_card(self, email_data: Dict[str, Any]):
        """Add an EmailCard to the triage area."""
        # Hide empty state if first card
        if self._email_cards:
            pass
        else:
            self.email_empty_label.pack_forget()
        
        card = EmailCard(
            self.email_cards_container,
            email_data,
            on_process=self._on_process_email_card,
            on_view=self._on_view_email_card
        )
        card.pack(fill=X, padx=5, pady=3)
        self._email_cards.append(card)
        
        return card
    
    def _clear_email_cards(self):
        """Remove all email cards from triage area."""
        for card in self._email_cards:
            card.destroy()
        self._email_cards = []
        self.email_empty_label.pack(expand=YES, pady=50)
    
    def _on_process_email_card(self, email_data: Dict):
        """Process an email: save attachments, extract links, run scanner."""
        subject = email_data.get('subject', 'N/A')[:50]
        self._email_log_message(f"▶️ Processando: {subject}", 'info')
        
        # Get the email object (stored during sync)
        email_obj = email_data.get('email_obj')
        if not email_obj:
            self._email_log_message("❌ Email object não encontrado", 'error')
            return
        
        # Process in background thread
        thread = threading.Thread(
            target=self._process_email_attachments,
            args=(email_obj, email_data),
            daemon=True
        )
        thread.start()
    
    def _process_email_attachments(self, email_obj, email_data: Dict):
        """Background processing of email attachments and links."""
        try:
            saved_files = []
            input_folder = Path(self.scanner.input_folder)
            
            # 1. Save PDF/XML attachments
            for attachment in email_obj.attachments:
                if attachment.filename.lower().endswith(('.pdf', '.xml')):
                    save_path = attachment.save_to(input_folder)
                    if save_path and save_path.exists():
                        saved_files.append(save_path)
                        self.root.after(0, lambda f=attachment.filename: 
                            self._email_log_message(f"  📄 Salvo: {f}", 'success'))
            
            # 2. Extract and download links (if no direct attachments)
            if not saved_files and LINK_EXTRACTOR_AVAILABLE and email_obj.html_body:
                self.root.after(0, lambda: self._email_log_message("  🔗 Buscando links...", 'info'))
                
                try:
                    extractor = LinkExtractor()
                    links = extractor.extract_links(email_obj.html_body)
                    
                    for link in links[:3]:  # Limit to top 3 links
                        if link.confidence >= 0.5:
                            self.root.after(0, lambda u=link.url[:40]: 
                                self._email_log_message(f"  🔗 Baixando: {u}...", 'info'))
                            
                            downloaded = download_link_sync(link.url, input_folder)
                            if downloaded and downloaded.exists():
                                saved_files.append(downloaded)
                                self.root.after(0, lambda f=downloaded.name: 
                                    self._email_log_message(f"  ✅ Link baixado: {f}", 'success'))
                except Exception as e:
                    self.root.after(0, lambda err=str(e): 
                        self._email_log_message(f"  ⚠️ Erro nos links: {err}", 'warning'))
            
            # 3. Process saved files with scanner
            if saved_files:
                self.root.after(0, lambda n=len(saved_files): 
                    self._email_log_message(f"  🔄 Processando {n} arquivo(s)...", 'info'))
                
                doc_ids = []
                for file_path in saved_files:
                    try:
                        ids = self.scanner.process_file(file_path)
                        doc_ids.extend(ids)
                    except Exception as e:
                        self.root.after(0, lambda f=file_path.name, err=str(e): 
                            self._email_log_message(f"  ❌ Erro em {f}: {err}", 'error'))
                
                # Update stats
                if doc_ids:
                    self.root.after(0, lambda: self._refresh_document_list())
                    self.root.after(0, lambda n=len(doc_ids): 
                        self._email_log_message(f"  ✅ {n} documento(s) importados!", 'success'))
                    
                    # Apply label to email (mark as processed)
                    if GMAIL_AVAILABLE:
                        try:
                            connector = GmailConnector()
                            if connector.authenticate():
                                connector.apply_label(email_obj.id, 'processed')
                        except:
                            pass
            else:
                self.root.after(0, lambda: 
                    self._email_log_message("  ⚠️ Nenhum arquivo encontrado para processar", 'warning'))
            
        except Exception as e:
            logger.error(f"Email processing error: {e}")
            self.root.after(0, lambda err=str(e): 
                self._email_log_message(f"❌ Erro: {err}", 'error'))
    
    def _on_view_email_card(self, email_data: Dict):
        """Show detailed email preview dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"📧 {email_data.get('subject', 'Email')[:50]}")
        dialog.geometry("700x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center on parent
        dialog.geometry(f"+{self.root.winfo_x() + 100}+{self.root.winfo_y() + 50}")
        
        # Main container
        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill=BOTH, expand=YES)
        
        # Header section
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=X, pady=(0, 10))
        
        # Status indicator
        status = email_data.get('status', 'pending')
        is_scheduled = email_data.get('is_scheduled', False)
        
        status_colors = {'ready': '🟢', 'warning': '🟡', 'error': '🔴', 'pending': '🔵'}
        status_icon = status_colors.get(status, '⚪')
        
        if is_scheduled:
            status_icon = '⚠️'
            status_text = "AGENDAMENTO DETECTADO"
            status_style = 'warning'
        else:
            status_text = status.upper()
            status_style = 'secondary'
        
        ttk.Label(
            header_frame,
            text=f"{status_icon} {status_text}",
            font=('Helvetica', 11, 'bold'),
            bootstyle=status_style
        ).pack(anchor=W)
        
        # Subject
        ttk.Label(
            header_frame,
            text=email_data.get('subject', 'Sem assunto'),
            font=('Helvetica', 14, 'bold'),
            wraplength=650
        ).pack(anchor=W, pady=(5, 0))
        
        # Metadata
        meta_frame = ttk.Frame(main_frame)
        meta_frame.pack(fill=X, pady=(0, 10))
        
        sender = email_data.get('sender', 'Desconhecido')
        date = email_data.get('date', '')
        ttk.Label(meta_frame, text=f"De: {sender}", bootstyle='secondary').pack(anchor=W)
        ttk.Label(meta_frame, text=f"Data: {date}", bootstyle='secondary').pack(anchor=W)
        
        # Content types
        content_types = email_data.get('content_types', [])
        if content_types:
            content_icons = {'pdf': '📄 PDF', 'xml': '📋 XML', 'link': '🔗 Link', 'attachment': '📎 Anexo'}
            content_str = " | ".join([content_icons.get(ct, ct) for ct in content_types])
            ttk.Label(meta_frame, text=f"Conteúdo: {content_str}", bootstyle='info').pack(anchor=W, pady=(5, 0))
        
        # Separator
        ttk.Separator(main_frame, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        # Body preview (scrollable)
        body_label = ttk.Label(main_frame, text="📝 Corpo do Email:", font=('Helvetica', 10, 'bold'))
        body_label.pack(anchor=W)
        
        body_frame = ttk.Frame(main_frame)
        body_frame.pack(fill=BOTH, expand=YES, pady=(5, 10))
        
        body_text = tk.Text(body_frame, wrap=tk.WORD, height=10, font=('Consolas', 9))
        body_scroll = ttk.Scrollbar(body_frame, orient=VERTICAL, command=body_text.yview)
        body_text.configure(yscrollcommand=body_scroll.set)
        
        body_scroll.pack(side=RIGHT, fill=Y)
        body_text.pack(side=LEFT, fill=BOTH, expand=YES)
        
        # Get email body
        email_obj = email_data.get('email_obj')
        if email_obj:
            try:
                body = getattr(email_obj, 'text_body', None) or getattr(email_obj, 'snippet', '') or ''
                if not body and hasattr(email_obj, 'html_body'):
                    # Strip HTML tags for preview
                    import re
                    html_body = email_obj.html_body or ''
                    body = re.sub(r'<[^>]+>', ' ', html_body)
                    body = re.sub(r'\s+', ' ', body).strip()[:2000]
            except:
                body = email_data.get('summary', 'Corpo não disponível')
        else:
            body = email_data.get('summary', 'Corpo não disponível')
        
        body_text.insert('1.0', body[:3000] if body else 'Sem conteúdo de texto')
        body_text.configure(state=DISABLED)
        
        # Attachments section
        if email_obj and hasattr(email_obj, 'attachments') and email_obj.attachments:
            attach_label = ttk.Label(main_frame, text=f"📎 Anexos ({len(email_obj.attachments)}):", font=('Helvetica', 10, 'bold'))
            attach_label.pack(anchor=W, pady=(5, 0))
            
            for att in email_obj.attachments[:5]:
                filename = getattr(att, 'filename', 'anexo')
                size = getattr(att, 'size', 0)
                size_str = f"{size/1024:.1f} KB" if size > 0 else ""
                ttk.Label(main_frame, text=f"  • {filename} {size_str}", bootstyle='secondary').pack(anchor=W)
        
        # Scheduling warning
        if is_scheduled:
            warning_frame = ttk.Frame(main_frame, bootstyle='warning')
            warning_frame.pack(fill=X, pady=10)
            ttk.Label(
                warning_frame,
                text="⚠️ ATENÇÃO: Este email pode conter um AGENDAMENTO de pagamento, não um comprovante!",
                font=('Helvetica', 10, 'bold'),
                bootstyle='warning',
                wraplength=650
            ).pack(padx=10, pady=10)
        
        # Action buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=X, pady=(10, 0))
        
        ttk.Button(
            btn_frame,
            text="▶ Processar",
            command=lambda: [dialog.destroy(), self._on_process_email_card(email_data)],
            bootstyle='success'
        ).pack(side=LEFT, padx=(0, 10))
        
        ttk.Button(
            btn_frame,
            text="Fechar",
            command=dialog.destroy,
            bootstyle='secondary-outline'
        ).pack(side=RIGHT)
    
    def _email_log_message(self, message: str, level: str = 'info'):
        """Add a message to the email log console."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        self.email_log.insert(END, f"[{timestamp}] ", 'timestamp')
        self.email_log.insert(END, f"{message}\n", level)
        self.email_log.see(END)
    
    def _on_start_email_monitor(self):
        """Start email monitoring pipeline."""
        if self._email_thread and self._email_thread.is_alive():
            self._email_log_message("Pipeline já está em execução!", 'warning')
            return
        
        # Update UI
        self.btn_start_email.config(state=DISABLED)
        self.btn_stop_email.config(state=NORMAL)
        
        self._email_log_message("Iniciando pipeline de email...", 'info')
        
        # Start pipeline in background thread
        self._email_thread = threading.Thread(target=self._run_email_pipeline, daemon=True)
        self._email_thread.start()
    
    def _on_stop_email_monitor(self):
        """Stop email monitoring pipeline."""
        if self._email_pipeline:
            self._email_pipeline.stop()
            self._email_log_message("Solicitando parada do pipeline...", 'warning')
        
        self.btn_start_email.config(state=NORMAL)
        self.btn_stop_email.config(state=DISABLED)
    
    def _run_email_pipeline(self):
        """Run email pipeline in background thread."""
        try:
            from email_pipeline import EmailPipeline, ProcessingStatus
            
            def progress_callback(message, current, total):
                self.root.after(0, lambda: self._email_log_message(message, 'info'))
                if total > 0:
                    stats_text = f"Emails: {current}/{total}"
                    self.root.after(0, lambda: self.lbl_email_stats.config(text=stats_text))
            
            self._email_pipeline = EmailPipeline(
                db=self.db,
                use_cloud_ai=True,
                progress_callback=progress_callback
            )
            
            results = self._email_pipeline.process_pending_emails(max_emails=20)
            
            # Log results
            success_count = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
            failed_count = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
            tokens_used = sum(r.gemini_tokens_used for r in results)
            
            self._gemini_total_tokens += tokens_used
            
            # Update UI
            self.root.after(0, lambda: self._email_log_message(
                f"Pipeline concluído: {success_count} sucesso, {failed_count} falhas",
                'success' if failed_count == 0 else 'warning'
            ))
            self.root.after(0, lambda: self.lbl_gemini_tokens.config(
                text=f"Tokens Gemini: {self._gemini_total_tokens:,}"
            ))
            
            # Log individual results
            for result in results[:10]:  # Show first 10
                status_icon = {
                    ProcessingStatus.SUCCESS: "✅",
                    ProcessingStatus.FAILED: "❌",
                    ProcessingStatus.SKIPPED: "⏭️",
                    ProcessingStatus.DUPLICATE: "🔄"
                }.get(result.status, "❓")
                
                level = 'success' if result.status == ProcessingStatus.SUCCESS else 'error'
                if result.gemini_tokens_used > 0:
                    level = 'gemini'
                
                self.root.after(0, lambda r=result, i=status_icon, l=level: 
                    self._email_log_message(f"{i} {r.subject[:50]}...", l))
            
            # Refresh document list
            self.root.after(0, self._refresh_all)
            
        except ImportError as e:
            self.root.after(0, lambda: self._email_log_message(
                f"Módulo não encontrado: {e}. Instale as dependências do Project Cyborg.",
                'error'
            ))
        except Exception as e:
            self.root.after(0, lambda: self._email_log_message(f"Erro: {e}", 'error'))
            logger.exception("Email pipeline error")
        finally:
            self.root.after(0, lambda: self.btn_start_email.config(state=NORMAL))
            self.root.after(0, lambda: self.btn_stop_email.config(state=DISABLED))
    
    def _on_gmail_sync_click(self):
        """Handle Gmail sync button click - fetches emails and populates cards."""
        # Switch to Email Monitor tab
        try:
            for tab_id in self.notebook.tabs():
                tab_text = self.notebook.tab(tab_id, 'text')
                if 'Email' in tab_text or '📧' in tab_text:
                    self.notebook.select(tab_id)
                    break
        except Exception as e:
            logger.warning(f"Could not switch to email tab: {e}")
        
        # Check if Gmail is available
        if not GMAIL_AVAILABLE:
            self._email_log_message("⚠️ GmailConnector não disponível. Verifique credentials.json", 'error')
            Messagebox.show_error(
                "Gmail não disponível", 
                "O módulo GmailConnector não está disponível.\n\nVerifique se credentials.json existe.",
                parent=self.root
            )
            return
        
        # Check if already syncing
        if self._email_thread and self._email_thread.is_alive():
            self._email_log_message("⏳ Sincronização já em andamento...", 'warning')
            return
        
        # Clear existing cards
        self._clear_email_cards()
        self._email_log_message("🔄 Iniciando sincronização com Gmail...", 'info')
        
        # Disable sync button
        self.btn_sync_now.config(state=DISABLED)
        
        # Start background thread
        self._email_thread = threading.Thread(target=self._run_gmail_sync, daemon=True)
        self._email_thread.start()
    
    def _run_gmail_sync(self):
        """Background thread to fetch and process emails from Gmail."""
        try:
            # Initialize connector
            connector = GmailConnector()
            
            # Authenticate (may open browser)
            self.root.after(0, lambda: self._email_log_message("🔐 Autenticando com Gmail...", 'info'))
            
            if not connector.authenticate():
                self.root.after(0, lambda: self._email_log_message("❌ Falha na autenticação", 'error'))
                return
            
            self.root.after(0, lambda: self._email_log_message("✅ Autenticado!", 'success'))
            
            # Fetch pending emails
            self.root.after(0, lambda: self._email_log_message("📥 Buscando emails pendentes...", 'info'))
            
            emails = list(connector.fetch_pending_emails(days_lookback=7, max_results=20))
            
            email_count = len(emails)
            self.root.after(0, lambda c=email_count: self._email_log_message(f"📧 {c} emails encontrados", 'success'))
            self.root.after(0, lambda c=email_count: self.lbl_email_stats.config(text=f"Emails: {c} | Docs: 0"))
            
            # Process each email and create cards
            for i, email in enumerate(emails):
                # Determine content types
                content_types = []
                if email.has_pdf_attachment():
                    content_types.append('pdf')
                if email.has_xml_attachment():
                    content_types.append('xml')
                if email.html_body and LINK_EXTRACTOR_AVAILABLE:
                    # Check for financial links
                    try:
                        extractor = LinkExtractor()
                        links = extractor.extract_links(email.html_body)
                        if links:
                            content_types.append('link')
                    except:
                        pass
                if email.attachments:
                    content_types.append('attachment')
                
                # Determine status
                status = 'pending'
                if email.has_pdf_attachment() or email.has_xml_attachment():
                    status = 'ready'
                
                # Check for scheduling keywords (BB)
                is_scheduled = False
                full_text = f"{email.subject} {email.html_body or ''}"
                scheduling_keywords = ['AGENDAMENTO', 'AGENDADO', 'PREVISÃO DE DÉBITO', 'PAGAMENTO AGENDADO']
                for kw in scheduling_keywords:
                    if kw.lower() in full_text.lower():
                        is_scheduled = True
                        status = 'scheduled'
                        break
                
                # Create email data for card
                email_data = {
                    'id': email.id,
                    'subject': email.subject,
                    'sender': email.sender,
                    'date': email.date,
                    'content_types': content_types,
                    'status': status,
                    'is_scheduled': is_scheduled,
                    'summary': f"{len(email.attachments)} anexos" if email.attachments else "",
                    'email_obj': email  # Store reference for processing
                }
                
                # Add card to UI (must be done in main thread)
                self.root.after(0, lambda data=email_data: self._add_email_card(data))
                
                # Log progress
                log_msg = f"  [{i+1}/{len(emails)}] {email.subject[:40]}..."
                self.root.after(0, lambda msg=log_msg: self._email_log_message(msg, 'info'))
            
            # Final status
            self.root.after(0, lambda: self._email_log_message("✅ Sincronização concluída!", 'success'))
            
        except Exception as e:
            logger.error(f"Gmail sync error: {e}")
            error_msg = str(e)  # Capture the error message before lambda
            self.root.after(0, lambda msg=error_msg: self._email_log_message(f"❌ Erro: {msg}", 'error'))
        finally:
            # Re-enable sync button
            self.root.after(0, lambda: self.btn_sync_now.config(state=NORMAL))
    
    # ==========================================================================
    # CONCILIATION TAB (Sprint 4)
    # ==========================================================================
    
    def _build_conciliation_tab(self):
        """Build the Conciliation tab for viewing document groups."""
        # Control buttons frame
        control_frame = ttk.Frame(self.conciliation_tab)
        control_frame.pack(fill=X, padx=5, pady=5)
        
        ttk.Button(
            control_frame,
            text="🔄 Atualizar Grupos",
            command=self._refresh_conciliation_groups,
            bootstyle="info"
        ).pack(side=LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="📦 Arquivar Grupo",
            command=self._on_archive_selected_group,
            bootstyle="success"
        ).pack(side=LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="🔗 Auto-Match Todos",
            command=self._on_auto_match_all,
            bootstyle="warning"
        ).pack(side=LEFT, padx=5)
        
        # Stats label
        self.conciliation_stats = ttk.Label(
            control_frame, 
            text="Grupos: 0 | Pendentes: 0 | Completos: 0",
            bootstyle="info"
        )
        self.conciliation_stats.pack(side=RIGHT, padx=5)
        
        # TreeView for groups
        columns = ("link_id", "status", "tipo_docs", "fornecedor", "valor", "data_master")
        self.groups_tree = ttk.Treeview(
            self.conciliation_tab, 
            columns=columns, 
            show="headings", 
            bootstyle="info",
            selectmode="browse"
        )
        
        self.groups_tree.heading("link_id", text="Grupo")
        self.groups_tree.heading("status", text="Status")
        self.groups_tree.heading("tipo_docs", text="Documentos")
        self.groups_tree.heading("fornecedor", text="Fornecedor")
        self.groups_tree.heading("valor", text="Valor")
        self.groups_tree.heading("data_master", text="Data Soberana")
        
        self.groups_tree.column("link_id", width=60, anchor=CENTER)
        self.groups_tree.column("status", width=100, anchor=CENTER)
        self.groups_tree.column("tipo_docs", width=120, anchor=CENTER)
        self.groups_tree.column("fornecedor", width=180)
        self.groups_tree.column("valor", width=100, anchor=E)
        self.groups_tree.column("data_master", width=120, anchor=CENTER)
        
        scrollbar = ttk.Scrollbar(self.conciliation_tab, orient=VERTICAL, command=self.groups_tree.yview)
        self.groups_tree.configure(yscrollcommand=scrollbar.set)
        
        self.groups_tree.pack(side=LEFT, fill=BOTH, expand=YES, padx=5, pady=5)
        scrollbar.pack(side=RIGHT, fill=Y, pady=5)
        
        # Double-click to show group details
        self.groups_tree.bind("<Double-1>", self._on_group_double_click)
    
    def _refresh_conciliation_groups(self):
        """Refresh the groups list in the conciliation tab."""
        # Clear current items
        for item in self.groups_tree.get_children():
            self.groups_tree.delete(item)
        
        try:
            groups = self.group_matcher.get_all_groups()
            unmatched = self.group_matcher.get_unmatched_documents()
            
            pending = 0
            complete = 0
            
            for group in groups:
                # Build doc types string
                doc_types = []
                if group.has_nfe:
                    doc_types.append("📄 NFe")
                if group.has_boleto:
                    doc_types.append("💰 Boleto")
                if group.has_comprovante:
                    doc_types.append("✅ Comp")
                tipos_str = " | ".join(doc_types) if doc_types else "❓ Vazio"
                
                # Get supplier from first document
                supplier = ""
                valor = 0
                for doc in group.documents:
                    if doc.supplier_clean:
                        supplier = doc.supplier_clean[:25]
                    if doc.amount_cents > 0:
                        valor = doc.amount_cents
                
                # Format value
                valor_str = f"R$ {valor / 100:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                
                # Status with emoji
                if group.is_complete:
                    status = "✅ Completo"
                    complete += 1
                elif group.is_partially_matched:
                    status = "🟡 Parcial"
                    pending += 1
                else:
                    status = "🔴 Pendente"
                    pending += 1
                
                # Data soberana
                master_date = group.master_date or "—"
                if master_date != "—":
                    master_date = self._iso_to_br(master_date)
                
                self.groups_tree.insert("", END, values=(
                    group.link_id,
                    status,
                    tipos_str,
                    supplier,
                    valor_str,
                    master_date
                ))
            
            # Update stats
            self.conciliation_stats.config(
                text=f"Grupos: {len(groups)} | Pendentes: {pending} | Completos: {complete} | Órfãos: {len(unmatched)}"
            )
            
            logger.info(f"Refreshed conciliation: {len(groups)} groups, {len(unmatched)} orphans")
            
        except Exception as e:
            logger.error(f"Error refreshing conciliation groups: {e}")
            self._set_status(f"Erro ao carregar grupos: {e}")
    
    def _on_group_double_click(self, event):
        """Handle double-click on a group to show details."""
        selection = self.groups_tree.selection()
        if not selection:
            return
        
        values = self.groups_tree.item(selection[0])['values']
        if values:
            link_id = values[0]
            group = self.group_matcher.get_group_by_link(link_id)
            
            if group:
                # Build details message
                details = f"Grupo #{link_id}\\n\\n"
                details += f"Status: {group.status}\\n"
                details += f"Data Soberana: {group.master_date or 'N/A'}\\n\\n"
                details += "Documentos:\\n"
                
                for doc in group.documents:
                    valor = f"R$ {doc.amount_cents / 100:.2f}" if doc.amount_cents else "—"
                    details += f"  • {doc.doc_type}: {doc.supplier_clean or 'N/A'} ({valor})\\n"
                
                Messagebox.showinfo(f"Grupo #{link_id}", details, parent=self.root)
    
    def _on_archive_selected_group(self):
        """Archive the selected group."""
        selection = self.groups_tree.selection()
        if not selection:
            Messagebox.show_warning("Selecione um grupo para arquivar.", "Aviso", parent=self.root)
            return
        
        values = self.groups_tree.item(selection[0])['values']
        if values:
            link_id = values[0]
            
            result = Messagebox.yesno(
                message=f"Arquivar grupo #{link_id}?\n\nOs documentos serão removidos da lista principal.",
                title="Arquivar Grupo",
                parent=self.root
            )
            
            if result == "Yes":
                count = self.db.archive_group(link_id)
                self._set_status(f"✅ Grupo #{link_id} arquivado ({count} documentos)")
                self._refresh_conciliation_groups()
                self._refresh_document_list()
    
    def _on_auto_match_all(self):
        """Attempt to auto-match all unmatched documents."""
        unmatched = self.group_matcher.get_unmatched_documents()
        
        if not unmatched:
            Messagebox.showinfo("Info", "Nenhum documento sem grupo.", parent=self.root)
            return
        
        result = Messagebox.yesno(
            title="Auto-Match",
            message=f"Tentar auto-match para {len(unmatched)} documentos?\\n\\n"
                    "Isso buscará matches automáticos por valor e fornecedor.",
            parent=self.root
        )
        
        if result == "Yes":
            matched = 0
            for doc in unmatched:
                link_id = self.group_matcher.auto_match_document(doc.doc_id)
                if link_id:
                    matched += 1
            
            self._set_status(f"✅ Auto-match: {matched}/{len(unmatched)} documentos vinculados")
            self._refresh_conciliation_groups()
            self._refresh_document_list()
    
    def _build_status_bar(self):
        status_bar = ttk.Frame(self.main_container)
        status_bar.pack(fill=X, pady=(10, 0))
        
        self.stats_frame = ttk.Frame(status_bar)
        self.stats_frame.pack(side=LEFT)
        
        self.lbl_total = ttk.Label(self.stats_frame, text="Total: 0", bootstyle="inverse-secondary")
        self.lbl_total.pack(side=LEFT, padx=(0, 15))
        
        self.lbl_islands = ttk.Label(self.stats_frame, text="Ilhas: 0", bootstyle="inverse-info")
        self.lbl_islands.pack(side=LEFT, padx=(0, 15))
        
        self.lbl_reconciled = ttk.Label(self.stats_frame, text="Reconciliados: 0", bootstyle="inverse-success")
        self.lbl_reconciled.pack(side=LEFT, padx=(0, 15))
        
        self.lbl_orphans = ttk.Label(self.stats_frame, text="Orfaos: 0", bootstyle="inverse-warning")
        self.lbl_orphans.pack(side=LEFT, padx=(0, 15))
        
        ttk.Label(status_bar, text="💡 Atalhos: Ctrl+A (selec. todos) | Delete (excluir) | F5 (atualizar) | Ctrl+E (editar)", font=("Helvetica", 8), bootstyle="secondary").pack(side=LEFT, padx=20)
        
        self.lbl_status = ttk.Label(status_bar, text="Pronto", bootstyle="secondary")
        self.lbl_status.pack(side=RIGHT)
        
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", bootstyle="success-striped")
    
    # ==========================================================================
    # EXIBICAO DE DETALHES
    # ==========================================================================
    
    def _show_no_selection(self):
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        frame = ttk.Frame(self.details_frame)
        frame.pack(expand=YES, fill=BOTH, padx=20, pady=50)
        
        ttk.Label(frame, text="Selecione um documento", font=("Helvetica", 12), bootstyle="secondary").pack()
        ttk.Label(frame, text="ou arraste PDFs para esta janela", bootstyle="secondary").pack(pady=5)
        
        self.preview_label.configure(image='', text="Nenhum documento")
        self.graph_text.delete(1.0, END)
    
    def _show_document_details(self, doc_id: int):
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT 
                d.id, d.file_hash, d.original_path, d.doc_type, 
                d.entity_tag, d.status, d.created_at, d.page_start, d.page_end,
                d.access_key, d.doc_number,
                t.amount_cents, t.supplier_clean, t.emission_date, t.due_date,
                t.payment_date, t.is_scheduled, t.sisbb_auth
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE d.id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if not row:
            self._show_no_selection()
            return
        
        columns = [col[0] for col in cursor.description]
        doc = dict(zip(columns, row))
        
        # Reset pagination
        self.current_preview_page = 0
        self.total_pages = 1  # Will be updated by preview
        
        container = ttk.Frame(self.details_frame)
        container.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        # Cabecalho
        header = ttk.Frame(container)
        header.pack(fill=X, pady=(0, 15))
        
        doc_type = doc.get('doc_type', 'UNKNOWN')
        icon = DOC_ICONS.get(doc_type, "❓")
        
        ttk.Label(header, text=f"{icon} {doc_type}", font=("Helvetica", 16, "bold"), bootstyle="primary").pack(side=LEFT)
        
        entity = doc.get('entity_tag', '')
        if entity:
            style = "success" if entity == "VG" else "info"
            ttk.Label(header, text=entity, font=("Helvetica", 14, "bold"), bootstyle=style).pack(side=RIGHT)
        

        ttk.Separator(container, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        # Tabela de Parcelas (se disponivel)
        installments = self.db.get_installments_by_nfe(doc_id)
        if installments:
            inst_frame = ttk.Labelframe(container, text=f" Parcelas ({len(installments)}) ", padding=5, bootstyle="info")
            inst_frame.pack(fill=X, pady=(0, 10))
            
            # Simple header
            ttk.Label(inst_frame, text="#", font=("Consolas", 9, "bold")).grid(row=0, column=0, padx=5)
            ttk.Label(inst_frame, text="Vencimento", font=("Consolas", 9, "bold")).grid(row=0, column=1, padx=5)
            ttk.Label(inst_frame, text="Valor", font=("Consolas", 9, "bold")).grid(row=0, column=2, padx=5)
            ttk.Label(inst_frame, text="Status", font=("Consolas", 9, "bold")).grid(row=0, column=3, padx=5)
            
            for idx, inst in enumerate(installments):
                r = idx + 1
                seq = inst['seq_num']
                dt = self._iso_to_br(inst['due_date'])
                val = f"R$ {SRDADatabase.cents_to_display(inst['amount_cents'])}"
                status = "✅" if inst['reconciled'] else "⭕"
                
                ttk.Label(inst_frame, text=f"{seq:02d}", font=("Consolas", 9)).grid(row=r, column=0, padx=5)
                ttk.Label(inst_frame, text=dt, font=("Consolas", 9)).grid(row=r, column=1, padx=5)
                ttk.Label(inst_frame, text=val, font=("Consolas", 9)).grid(row=r, column=2, padx=5)
                ttk.Label(inst_frame, text=status, font=("Consolas", 9)).grid(row=r, column=3, padx=5)
            
            ttk.Separator(container, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        # Informacoes
        info_frame = ttk.Frame(container)
        info_frame.pack(fill=X)
        
        row_num = 0
        def add_row(label, value, highlight=False):
            nonlocal row_num
            ttk.Label(info_frame, text=label, font=("Helvetica", 10, "bold"), bootstyle="secondary").grid(row=row_num, column=0, sticky=W, pady=3)
            style = "success" if highlight else "default"
            ttk.Label(info_frame, text=value or "-", font=("Helvetica", 10), bootstyle=style, wraplength=250).grid(row=row_num, column=1, sticky=W, padx=(10, 0), pady=3)
            row_num += 1
        
        add_row("ID:", str(doc.get('id', '')))
        add_row("Status:", doc.get('status', ''))
        
        amount = doc.get('amount_cents', 0)
        if amount:
            add_row("Valor:", f"R$ {SRDADatabase.cents_to_display(amount)}", highlight=True)
        
        add_row("Fornecedor:", doc.get('supplier_clean', ''))
        
        if doc.get('payment_date'):
            add_row("📅 Data Pagamento:", doc['payment_date'], highlight=True)
        if doc.get('due_date'):
            add_row("Data Vencimento:", doc['due_date'])
        if doc.get('emission_date'):
            add_row("Data Emissao:", doc['emission_date'])
        
        if doc.get('sisbb_auth'):
            add_row("🔐 SISBB:", doc['sisbb_auth'], highlight=True)
        
        if doc.get('is_scheduled'):
            ttk.Label(container, text="⚠️ AGENDAMENTO (nao e pagamento confirmado)", bootstyle="warning").pack(anchor=W, pady=10)
        
        # Preview
        path = doc.get('original_path', '')
        
        # Titulo Preview e Botoes
        prev_header = ttk.Frame(container)
        prev_header.pack(fill=X, pady=(15, 5))
        ttk.Label(prev_header, text="Visualização", font=("Helvetica", 10, "bold"), bootstyle="secondary").pack(side=LEFT)
        
        # Pagination Controls
        self.btn_prev_page = ttk.Button(prev_header, text="<", width=3, bootstyle="secondary-outline", command=lambda: self._change_page(-1))
        self.btn_prev_page.pack(side=RIGHT, padx=(5, 0))
        
        self.lbl_page_info = ttk.Label(prev_header, text="1/1", font=("Consolas", 9))
        self.lbl_page_info.pack(side=RIGHT, padx=(5, 5))
        
        self.btn_next_page = ttk.Button(prev_header, text=">", width=3, bootstyle="secondary-outline", command=lambda: self._change_page(1))
        self.btn_next_page.pack(side=RIGHT)
        
        # Zoom Controls (NEW)
        ttk.Separator(prev_header, orient=VERTICAL).pack(side=RIGHT, fill=Y, padx=8)
        
        ttk.Button(prev_header, text="−", width=2, bootstyle="secondary-outline", 
                   command=lambda: self._zoom_preview(-0.25)).pack(side=RIGHT)
        
        self.lbl_zoom_info = ttk.Label(prev_header, text="100%", font=("Consolas", 9))
        self.lbl_zoom_info.pack(side=RIGHT, padx=(3, 3))
        
        ttk.Button(prev_header, text="+", width=2, bootstyle="secondary-outline",
                   command=lambda: self._zoom_preview(0.25)).pack(side=RIGHT)
        
        if path and os.path.exists(path):
            self._show_pdf_preview(path)
        
        # Grafo
        self._show_document_graph(doc_id)
    
    def _change_page(self, delta: int):
        """Muda a pagina do preview."""
        if not self.selected_doc_id:
            return
            
        new_page = self.current_preview_page + delta
        if 0 <= new_page < self.total_pages:
            self.current_preview_page = new_page
            # Re-render preview
            cursor = self.db.connection.cursor()
            cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
            row = cursor.fetchone()
            if row and os.path.exists(row[0]):
                self._show_pdf_preview(row[0])
    
    def _zoom_preview(self, delta: float):
        """Zoom in or out on the PDF preview."""
        if not hasattr(self, '_preview_scale'):
            self._preview_scale = 1.0
        
        new_scale = self._preview_scale + delta
        if 0.5 <= new_scale <= 3.0:  # Limit zoom range
            self._preview_scale = new_scale
            
            # Re-render with new scale
            cursor = self.db.connection.cursor()
            cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
            row = cursor.fetchone()
            if row and os.path.exists(row[0]):
                self._show_pdf_preview(row[0])
                
            # Update zoom label
            if hasattr(self, 'lbl_zoom_info'):
                self.lbl_zoom_info.configure(text=f"{int(self._preview_scale * 100)}%")

    def _show_pdf_preview(self, pdf_path: str):
        try:
            doc = fitz.open(pdf_path)
            self.total_pages = len(doc)
            
            if self.total_pages == 0:
                doc.close()
                return
            
            # Ensure page in range
            if self.current_preview_page >= self.total_pages:
                self.current_preview_page = 0
            
            # Get scale factor (default 1.0)
            scale = getattr(self, '_preview_scale', 1.0)
            base_zoom = 0.8 * scale
                
            page = doc[self.current_preview_page]
            pix = page.get_pixmap(matrix=fitz.Matrix(base_zoom, base_zoom))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Calculate max size based on scale
            max_width = int(400 * scale)
            max_height = int(500 * scale)
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            self.preview_image = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_image, text="")
            
            # Update info
            self.lbl_page_info.configure(text=f"{self.current_preview_page + 1}/{self.total_pages}")
            
            # Update buttons state
            self.btn_prev_page.configure(state=NORMAL if self.current_preview_page > 0 else DISABLED)
            self.btn_next_page.configure(state=NORMAL if self.current_preview_page < self.total_pages - 1 else DISABLED)
            
            doc.close()
        except Exception as e:
            self.preview_label.configure(image='', text=f"Erro: {e}")
            logger.error(f"Erro no preview: {e}")
    
    def _show_document_graph(self, doc_id: int):
        self.graph_text.delete(1.0, END)
        
        matches = self.db.get_matches_by_document(doc_id)
        
        if not matches:
            self.graph_text.insert(END, "Este documento nao tem vinculos.\n\n")
            self.graph_text.insert(END, "Dica: Arraste outro documento sobre este\n")
            self.graph_text.insert(END, "para criar um vinculo manual.")
            return
        
        self.graph_text.insert(END, f"=== VINCULOS DO DOCUMENTO #{doc_id} ===\n\n")
        
        for match in matches:
            if match['parent_doc_id'] == doc_id:
                direction = "➜"
                other_id = match['child_doc_id']
                other_path = match['child_path']
            else:
                direction = "⟵"
                other_id = match['parent_doc_id']
                other_path = match['parent_path']
            
            confidence = match.get('confidence', 0) * 100
            match_type = match.get('match_type', 'UNKNOWN')
            
            self.graph_text.insert(END, f"{direction} Documento #{other_id}\n")
            self.graph_text.insert(END, f"   Arquivo: {Path(other_path).name}\n")
            self.graph_text.insert(END, f"   Tipo: {match_type} | Confianca: {confidence:.0f}%\n\n")
    
    def _on_show_graph_click(self):
        if self.selected_doc_id:
            self.notebook.select(2)  # Seleciona aba do grafo
    
    # ==========================================================================
    # ATUALIZACAO DE DADOS
    # ==========================================================================
    
    def _refresh_document_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        cursor = self.db.connection.cursor()
        
        query = """
            SELECT 
                d.id, d.doc_type, d.entity_tag, d.status,
                t.supplier_clean, t.amount_cents
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE 1=1
        """
        
        entity = self.filter_entity.get()
        if entity != "Todos":
            query += f" AND d.entity_tag = '{entity}'"
        
        doc_type = self.filter_type.get()
        if doc_type != "Todos":
            query += f" AND d.doc_type = '{doc_type}'"
        
        status = self.filter_status.get()
        if status != "Todos":
            query += f" AND d.status = '{status}'"
        
        query += " ORDER BY d.id DESC"
        
        cursor.execute(query)
        
        columns = [col[0] for col in cursor.description]
        
        for row in cursor.fetchall():
            doc = dict(zip(columns, row))
            
            doc_type = doc.get('doc_type', 'UNKNOWN')
            icon = DOC_ICONS.get(doc_type, "❓")
            
            entity = doc.get('entity_tag', '-') or '-'
            supplier = doc.get('supplier_clean', '-') or '-'
            status = doc.get('status', '-')
            
            amount = doc.get('amount_cents', 0)
            amount_str = f"R$ {SRDADatabase.cents_to_display(amount)}" if amount else "-"
            
            if len(supplier) > 20:
                supplier = supplier[:17] + "..."
            
            self.tree.insert("", END, values=(doc['id'], f"{icon} {doc_type}", entity, supplier, amount_str, status), tags=(status,))
        
        for status, style in STATUS_COLORS.items():
            self.tree.tag_configure(status, foreground=self._get_color(style))
    
    def _get_color(self, style: str) -> str:
        colors = {"primary": "#0d6efd", "secondary": "#6c757d", "success": "#198754", "danger": "#dc3545", "warning": "#ffc107", "info": "#0dcaf0"}
        return colors.get(style, "#ffffff")
    
    def _update_statistics(self):
        stats = self.db.get_statistics()
        
        total = stats.get('total_documents', 0)
        by_status = stats.get('by_status', {})
        reconciled = by_status.get('RECONCILED', 0) + by_status.get('RENAMED', 0)
        
        self.lbl_total.configure(text=f"Total: {total}")
        self.lbl_reconciled.configure(text=f"Reconciliados: {reconciled}")
        self.lbl_islands.configure(text=f"Ilhas: {len(self.current_islands)}")
        self.lbl_orphans.configure(text=f"Orfaos: {total - reconciled}")
    
    def _refresh_all(self):
        self._refresh_document_list()
        self._update_statistics()
        self._update_stats_dashboard()  # Premium: Update header stats
        self._set_status("Lista atualizada")
        self._show_toast("Lista atualizada ✅", 'success', 2000)
    
    # ==========================================================================
    # EVENTOS
    # ==========================================================================
    
    def _on_tree_selection_change(self, event):
        """Handle TreeView selection changes - supports multi-select."""
        selection = self.tree.selection()
        count = len(selection)
        
        if count == 0:
            self._show_no_selection()
        elif count == 1:
            # Single selection - show details
            self._on_document_select(event)
        else:
            # Multiple selection - show count in status
            self._set_status(f"📋 {count} itens selecionados (Delete para excluir, Ctrl+R para reprocessar)")
    
    def _on_document_select(self, event):
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        doc_id = item['values'][0]
        
        self.selected_doc_id = doc_id
        self._show_document_details(doc_id)
    
    def _on_add_files_click(self):
        """Add files with background processing and progress feedback."""
        files = filedialog.askopenfilenames(
            title="Selecione PDFs", 
            filetypes=[("PDF", "*.pdf"), ("XML", "*.xml"), ("Todos", "*.*")]
        )
        
        if not files:
            return
        
        # Process in background thread to avoid UI freeze
        def process_files_background():
            total = len(files)
            processed = 0
            errors = 0
            
            for f in files:
                try:
                    filename = Path(f).name
                    self.root.after(0, lambda fn=filename, i=processed, t=total: 
                        self._set_status(f"⏳ Processando [{i+1}/{t}]: {fn[:30]}..."))
                    
                    self.scanner.process_file(Path(f))
                    processed += 1
                    
                    # Show progress toast for each file
                    self.root.after(0, lambda fn=filename: 
                        self._show_toast(f"✅ {fn[:25]}", 'success', 1500))
                    
                except Exception as e:
                    errors += 1
                    filename = Path(f).name
                    self.root.after(0, lambda fn=filename, err=str(e): 
                        self._show_toast(f"❌ {fn[:20]}: {err[:30]}", 'error', 3000))
            
            # Final update
            self.root.after(0, lambda: self._refresh_all())
            
            if errors == 0:
                self.root.after(0, lambda p=processed: 
                    self._set_status(f"✅ {p} arquivo(s) importados com sucesso!"))
            else:
                self.root.after(0, lambda p=processed, e=errors: 
                    self._set_status(f"⚠️ {p} OK, {e} erros"))
        
        # Start thread
        self._set_status(f"⏳ Iniciando importação de {len(files)} arquivo(s)...")
        thread = threading.Thread(target=process_files_background, daemon=True)
        thread.start()
    
    def _on_scan_click(self):
        if self.is_processing:
            return
        
        folder = filedialog.askdirectory(title="Selecione pasta com PDFs", initialdir=".")
        if folder:
            self._run_in_thread(self._do_scan, folder)
    
    def _on_match_click(self):
        if self.is_processing:
            return
        self._run_in_thread(self._do_match)
    
    def _on_rename_click(self):
        """Rename files with output folder selection."""
        if self.is_processing:
            return
        
        # First, ask for output folder
        output_folder = filedialog.askdirectory(
            title="Selecione pasta de destino para os arquivos renomeados",
            initialdir="./Output"
        )
        
        if not output_folder:
            return  # User cancelled
        
        # Store the selected folder for use in _do_rename
        self._rename_output_folder = output_folder
        
        # Confirm with count
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM documentos WHERE status = 'RECONCILED'")
        count = cursor.fetchone()[0]
        
        if count == 0:
            Messagebox.showinfo("Info", "Nenhum documento reconciliado para renomear.", parent=self.root)
            return
        
        result = Messagebox.yesno(
            title="Confirmar Renomeação",
            message=f"Renomear {count} arquivo(s) reconciliados?\n\n"
                    f"📁 Destino: {output_folder}\n\n"
                    "⚠️ Arquivos serão COPIADOS (originais intactos).",
            parent=self.root
        )
        if result == "Yes":
            self._run_in_thread(self._do_rename)
    
    def _on_edit_click(self):
        if not self.selected_doc_id:
            return
        
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT d.id, d.doc_type, d.entity_tag, t.amount_cents, t.supplier_clean, t.due_date, t.payment_date
            FROM documentos d LEFT JOIN transacoes t ON d.id = t.doc_id WHERE d.id = ?
        """, (self.selected_doc_id,))
        
        row = cursor.fetchone()
        if not row:
            return
        
        # Build doc_data with additional metadata for Active Learning
        columns = [col[0] for col in cursor.description]
        doc_data = dict(zip(columns, row))
        doc_data['doc_id'] = self.selected_doc_id  # Required for corrections logging
        
        dialog = EditDialog(self.root, doc_data, db=self.db)
        self.root.wait_window(dialog)
        
        if dialog.result:
            self.db.update_transaction_fields(doc_id=self.selected_doc_id, **dialog.result)
            self._refresh_all()
            self._show_document_details(self.selected_doc_id)
            self._set_status("Dados atualizados")
    
    def _on_reprocess_click(self):
        if not self.selected_doc_id:
            return
        
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
        row = cursor.fetchone()
        if not row or not os.path.exists(row[0]):
            Messagebox.show_error("Erro", "Arquivo nao encontrado", parent=self.root)
            return
        
        if Messagebox.yesno("Reprocessar", "Excluir dados atuais e reprocessar?", parent=self.root) == "Yes":
            self._delete_document(self.selected_doc_id)
            self.scanner.process_file(Path(row[0]))
            self._refresh_all()
            self._set_status("Documento reprocessado")
    
    def _on_open_file_click(self):
        if not self.selected_doc_id:
            return
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
        row = cursor.fetchone()
        if row and os.path.exists(row[0]):
            os.startfile(row[0])
    
    def _on_open_folder_click(self):
        if not self.selected_doc_id:
            return
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
        row = cursor.fetchone()
        if row and os.path.exists(row[0]):
            os.startfile(os.path.dirname(row[0]))
    
    def _on_delete_click(self):
        if not self.selected_doc_id:
            return
        
        if Messagebox.yesno("Excluir", "Excluir documento do banco?\n\nArquivo original NAO sera apagado.", parent=self.root) == "Yes":
            self._delete_document(self.selected_doc_id)
            self.selected_doc_id = None
            self._show_no_selection()
            self._refresh_all()
            self._set_status("Documento excluido")
    
    def _delete_document(self, doc_id: int):
        cursor = self.db.connection.cursor()
        cursor.execute("DELETE FROM corrections_log WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM matches WHERE parent_doc_id = ? OR child_doc_id = ?", (doc_id, doc_id))
        cursor.execute("DELETE FROM duplicatas WHERE nfe_id = ? OR boleto_id = ?", (doc_id, doc_id))
        cursor.execute("DELETE FROM transacoes WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documentos WHERE id = ?", (doc_id,))
        self.db.connection.commit()
    
    # ==========================================================================
    # OPERACOES EM THREAD
    # ==========================================================================
    
    def _run_in_thread(self, func, *args):
        self.is_processing = True
        self._set_buttons_state(False)
        self._show_progress(True)
        
        def wrapper():
            try:
                func(*args)
            except Exception as e:
                self.root.after(0, lambda: Messagebox.show_error("Erro", str(e), parent=self.root))
            finally:
                self.root.after(0, self._on_operation_complete)
        
        threading.Thread(target=wrapper, daemon=True).start()
    
    def _on_operation_complete(self):
        self.is_processing = False
        self._set_buttons_state(True)
        self._show_progress(False)
        self._refresh_all()
    
    def _set_buttons_state(self, enabled: bool):
        state = NORMAL if enabled else DISABLED
        for btn in [self.btn_add, self.btn_scan, self.btn_process, self.btn_match, self.btn_rename]:
            btn.configure(state=state)
    
    def _show_progress(self, show: bool):
        if show:
            self.progress.pack(side=RIGHT, padx=(10, 0))
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.pack_forget()
    
    def _set_status(self, text: str):
        self.lbl_status.configure(text=text)
    
    # ==========================================================================
    # OPERACOES
    # ==========================================================================
    
    def _do_scan(self, folder: str):
        """Importa PDFs rapidamente SEM AI - apenas registra no banco."""
        self.root.after(0, lambda: self._set_status("� Importando PDFs..."))
        self.scanner.input_folder = Path(folder)
        
        def progress_callback(current, total, filename, stage):
            if stage == "nenhum_arquivo":
                msg = "⚠️ Nenhum PDF encontrado na pasta"
            elif stage == "importando":
                msg = f"� [{current}/{total}] Importando: {filename[:40]}..."
            elif stage == "ok":
                msg = f"✅ [{current}/{total}] Importado: {filename[:35]}"
            elif stage == "skip":
                msg = f"⏭️ [{current}/{total}] Já existe: {filename[:35]}"
            elif stage == "erro":
                msg = f"❌ [{current}/{total}] Erro: {filename[:40]}"
            else:
                msg = f"🔄 [{current}/{total}] {filename[:40]}"
            
            self.root.after(0, lambda m=msg: self._set_status(m))
        
        # Usa quick_import (SEM AI) em vez de process_all
        stats = self.scanner.quick_import(progress_callback=progress_callback)
        
        # Status final
        msg = f"✅ Importado: {stats['imported']} novos | {stats['skipped']} existentes"
        self.root.after(0, lambda: self._set_status(msg))
        
        # Prompt for immediate processing if new files found
        if stats['new_ids']:
            ids_to_process = stats['new_ids']
            if len(ids_to_process) > 0:
                self.root.after(0, lambda: self._prompt_processing(ids_to_process))

    def _prompt_processing(self, doc_ids: List[int]):
        """Prompt user to process imported files immediately."""
        msg = f"Importação concluída. Deseja processar (OCR/AI) os {len(doc_ids)} novos arquivos agora?"
        if Messagebox.show_question("Processar Agora?", msg, parent=self.root) == "Yes":
            self._run_in_thread(self._do_process_selected, doc_ids)
    

    def _do_match(self):
        """Reconcilia documentos com feedback visual usando MCMF."""
        self.root.after(0, lambda: self._set_status("🔗 Preparando dados..."))
        
        # 1. Fetch data
        docs_data = self.db.get_all_open_documents()
        nodes = []
        for d in docs_data:
            nodes.append(DocumentNode(
                id=d['id'],
                doc_type=d['doc_type'],
                amount_cents=d['amount_cents'] or 0,
                supplier=d['supplier_clean'],
                due_date=d['due_date'],
                payment_date=d['payment_date'],
                emission_date=d['emission_date'],
                entity_tag=d['entity_tag'],
                original_path=d['original_path']
            ))
            
        self.root.after(0, lambda: self._set_status(f"🧮 Maximizando fluxo (MCMF) em {len(nodes)} nós..."))
        
        # 2. Run MCMF
        islands = self.reconciler.reconcile(nodes)
        self.current_islands = islands
        
        self.root.after(0, lambda: self._set_status("💾 Persistindo grafo..."))
        
        # 3. Persist
        count_matches = 0
        try:
            for island in islands:
                for edge in island.edges:
                     # Edge is dict {source, target, weight}
                     s_id = edge['source']
                     t_id = edge['target']
                     
                     # Find nodes to check types
                     s_node = next((n for n in island.nodes if n.id == s_id), None)
                     t_node = next((n for n in island.nodes if n.id == t_id), None)
                     
                     if s_node and t_node:
                         is_p_s = s_node.doc_type in ['BOLETO', 'COMPROVANTE']
                         # is_p_t = t_node.doc_type in ['BOLETO', 'COMPROVANTE']
                         
                         if is_p_s:
                             self.db.insert_match(s_id, t_id, MatchType.EXACT, 1.0)
                         else:
                             self.db.insert_match(t_id, s_id, MatchType.EXACT, 1.0)
                         count_matches += 1
            
            self.db.connection.commit()
        except Exception as e:
             logger.error(f"Persistence error: {e}")
        

        msg = f"✅ Reconciliado: {len(islands)} ilhas | {count_matches} vínculos gerados pelo Solver"
        
        # 4. Post-Match Heuristic: Link Installments
        # If we matched Boleto to NFE, try to find matching installment
        try:
             # Iterate recently created matches (or all confirmed matches from this session)
             # Efficient way: Re-iterate islands with edges
             linked_count = 0
             for island in islands:
                for edge in island.edges:
                    s_id = edge['source']
                    t_id = edge['target']
                    s_node = next((n for n in island.nodes if n.id == s_id), None)
                    t_node = next((n for n in island.nodes if n.id == t_id), None)
                    
                    if not s_node or not t_node: continue

                    # Identify Parent (Boleto) and Child (NFE)
                    boleto_node = None
                    nfe_node = None
                    
                    if s_node.doc_type in ['BOLETO', 'COMPROVANTE'] and t_node.doc_type in ['NFE', 'NFSE']:
                        boleto_node = s_node
                        nfe_node = t_node
                    elif t_node.doc_type in ['BOLETO', 'COMPROVANTE'] and s_node.doc_type in ['NFE', 'NFSE']:
                        boleto_node = t_node
                        nfe_node = s_node
                        
                    if boleto_node and nfe_node:
                        # Try to link installment
                        if self.db.link_installment_to_boleto(nfe_node.id, boleto_node.id, boleto_node.amount_cents):
                            linked_count += 1
             
             if linked_count > 0:
                 msg += f" | {linked_count} parcelas vinculadas"
                 
        except Exception as e:
            logger.error(f"Installment linking error: {e}")

        self.root.after(0, lambda: self._set_status(msg))
        self.root.after(0, self._refresh_all)
    
    def _do_rename(self):
        """Renomeia arquivos com feedback visual e pasta customizada."""
        output_folder = getattr(self, '_rename_output_folder', './Output')
        
        self.root.after(0, lambda: self._set_status(f"📋 Preparando operações de renomeação para {output_folder}..."))
        self.root.after(0, lambda: self._show_toast(f"📁 Destino: {Path(output_folder).name}", 'info', 2000))
        
        # Update renamer output directory if it has such attribute
        if hasattr(self.renamer, 'output_dir'):
            self.renamer.output_dir = Path(output_folder)
        
        result = self.renamer.run(dry_run=False, copy_mode=True)
        
        if result.failed > 0:
            msg = f"⚠️ Renomeação: {result.successful} OK | {result.failed} falhas | {result.skipped} ignorados"
            self.root.after(0, lambda: self._show_toast(msg, 'warning', 4000))
        else:
            msg = f"✅ Renomeado: {result.successful} arquivos copiados para {Path(output_folder).name}/"
            self.root.after(0, lambda: self._show_toast(msg, 'success', 3000))
        
        self.root.after(0, lambda: self._set_status(msg))
    
    # ==========================================================================
    # EXECUCAO
    # ==========================================================================
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
    
    def _on_closing(self):
        """Manipula fechamento da janela."""
        if self.is_processing:
            if Messagebox.show_question("Sair?", "Processamento em andamento. Deseja forçar saída?", parent=self.root) != "Yes":
                return
        
        self.cleanup()
        self.root.destroy()
        sys.exit(0)
    
    def cleanup(self):
        self.db.close()


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

def main():
    app = SRDAApplication()
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
