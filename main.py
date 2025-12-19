"""
SRDA-Rural - Interface Grafica Principal v2.0
==============================================
Sistema de Reconciliacao Documental e Automacao Rural

Melhorias v2.0:
- Menu de contexto (botao direito): Editar, Excluir, Reprocessar, Abrir Arquivo
- Dialog de edicao de dados inline
- Preview do PDF no painel de detalhes
- Arrastar e soltar arquivos PDF
- Confirmacoes antes de acoes destrutivas

Referencia: Automacao e Reconciliacao Financeira com IA.txt
"""

import os
import sys
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import tkinter as tk
from tkinter import simpledialog

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
try:
    from ttkbootstrap.scrolled import ScrolledFrame
except ImportError:
    from ttkbootstrap.widgets import ScrolledFrame
from ttkbootstrap.dialogs import Messagebox
try:
    from ttkbootstrap.tooltip import ToolTip
except ImportError:
    from ttkbootstrap.widgets import ToolTip
from tkinter import filedialog
import fitz  # PyMuPDF
from PIL import Image, ImageTk

# Importa modulos do sistema
from database import (
    SRDADatabase,
    DocumentType,
    DocumentStatus,
    EntityTag
)
from scanner import CognitiveScanner
from matcher import ReconciliationEngine
from renamer import DocumentRenamer


# ==============================================================================
# CONFIGURACOES
# ==============================================================================

APP_TITLE = "SRDA-Rural - Sistema de Reconciliacao Documental v2.0"
APP_VERSION = "2.0.0"
THEME = "darkly"  # Temas: darkly, superhero, solar, cyborg, vapor

# Cores de status
STATUS_COLORS = {
    "INGESTED": "info",
    "PARSED": "primary",
    "RECONCILED": "success",
    "RENAMED": "warning",
    "ERROR": "danger",
}

# Icones de tipo de documento
DOC_ICONS = {
    "NFE": "[NF]",
    "NFSE": "[NS]",
    "BOLETO": "[BO]",
    "COMPROVANTE": "[CP]",
    "UNKNOWN": "[??]",
}


# ==============================================================================
# DIALOG DE EDICAO
# ==============================================================================

class EditDialog(tk.Toplevel):
    """Dialog para edicao de dados de um documento."""
    
    def __init__(self, parent, doc_data: Dict[str, Any]):
        super().__init__(parent)
        self.result = None
        self.doc_data = doc_data
        
        self.title("Editar Documento")
        self.geometry("400x350")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Centraliza
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 200
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 175
        self.geometry(f"+{x}+{y}")
        
        self._build_ui()
        
        # Focus no primeiro campo
        self.entry_supplier.focus()
    
    def _build_ui(self):
        """Constroi a interface do dialog."""
        main = ttk.Frame(self, padding=20)
        main.pack(fill=BOTH, expand=YES)
        
        # Titulo
        ttk.Label(
            main,
            text="Editar Dados Extraidos",
            font=("Helvetica", 14, "bold"),
            bootstyle="primary"
        ).pack(anchor=W, pady=(0, 15))
        
        # Fornecedor
        ttk.Label(main, text="Fornecedor/Destinatario:").pack(anchor=W)
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
        due = self.doc_data.get('due_date', '')
        if due:
            # Converte ISO para BR
            try:
                parts = due.split("-")
                self.entry_due_date.insert(0, f"{parts[2]}/{parts[1]}/{parts[0]}")
            except:
                self.entry_due_date.insert(0, due)
        
        # Data de Pagamento
        ttk.Label(main, text="Data de Pagamento (DD/MM/AAAA):").pack(anchor=W)
        self.entry_payment_date = ttk.Entry(main, width=20)
        self.entry_payment_date.pack(anchor=W, pady=(2, 10))
        payment = self.doc_data.get('payment_date', '')
        if payment:
            try:
                parts = payment.split("-")
                self.entry_payment_date.insert(0, f"{parts[2]}/{parts[1]}/{parts[0]}")
            except:
                self.entry_payment_date.insert(0, payment)
        
        # Botoes
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=X, pady=(20, 0))
        
        ttk.Button(
            btn_frame,
            text="Cancelar",
            bootstyle="secondary",
            command=self.destroy
        ).pack(side=RIGHT, padx=(5, 0))
        
        ttk.Button(
            btn_frame,
            text="Salvar",
            bootstyle="success",
            command=self._on_save
        ).pack(side=RIGHT)
    
    def _on_save(self):
        """Salva as alteracoes."""
        # Valida e converte valor
        amount_str = self.entry_amount.get().replace(".", "").replace(",", ".")
        try:
            amount_cents = int(float(amount_str) * 100) if amount_str else 0
        except:
            amount_cents = 0
        
        # Converte datas BR para ISO
        def br_to_iso(date_br):
            if not date_br:
                return None
            try:
                parts = date_br.split("/")
                return f"{parts[2]}-{parts[1]}-{parts[0]}"
            except:
                return None
        
        self.result = {
            'supplier': self.entry_supplier.get().strip().upper(),
            'amount_cents': amount_cents,
            'due_date': br_to_iso(self.entry_due_date.get()),
            'payment_date': br_to_iso(self.entry_payment_date.get()),
        }
        self.destroy()


# ==============================================================================
# CLASSE PRINCIPAL: SRDAApplication
# ==============================================================================

class SRDAApplication:
    """
    Aplicacao principal do SRDA-Rural v2.0.
    
    Melhorias:
    - Menu de contexto (botao direito)
    - Edicao inline de dados
    - Preview do PDF
    - Drag & Drop de arquivos
    """
    
    def __init__(self):
        """Inicializa a aplicacao."""
        # Componentes do sistema
        self.db = SRDADatabase()
        self.scanner = CognitiveScanner(db=self.db)
        self.matcher = ReconciliationEngine(db=self.db)
        self.renamer = DocumentRenamer(db=self.db)
        
        # Janela principal
        self.root = ttk.Window(
            title=APP_TITLE,
            themename=THEME,
            size=(1300, 750),
            resizable=(True, True)
        )
        self.root.minsize(1000, 600)
        
        # Centraliza janela
        self._center_window()
        
        # Variaveis de estado
        self.selected_doc_id: Optional[int] = None
        self.is_processing = False
        self.preview_image = None  # Mantem referencia para nao ser garbage collected
        
        # Constroi interface
        self._build_ui()
        
        # Menu de contexto
        self._create_context_menu()
        
        # Configura Drag & Drop
        self._setup_drag_drop()
        
        # Carrega dados iniciais
        self._refresh_document_list()
        self._update_statistics()
    
    def _center_window(self):
        """Centraliza a janela na tela."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")
    
    # ==========================================================================
    # MENU DE CONTEXTO (BOTAO DIREITO)
    # ==========================================================================
    
    def _create_context_menu(self):
        """Cria o menu de contexto para a Treeview."""
        self.context_menu = tk.Menu(self.root, tearoff=0)
        
        self.context_menu.add_command(
            label="Editar Dados...",
            command=self._on_edit_click
        )
        self.context_menu.add_command(
            label="Reprocessar PDF",
            command=self._on_reprocess_click
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="Abrir Arquivo Original",
            command=self._on_open_file_click
        )
        self.context_menu.add_command(
            label="Abrir Pasta do Arquivo",
            command=self._on_open_folder_click
        )
        self.context_menu.add_separator()
        self.context_menu.add_command(
            label="Excluir do Banco de Dados",
            command=self._on_delete_click
        )
        
        # Bind para botao direito
        self.tree.bind("<Button-3>", self._show_context_menu)
    
    def _show_context_menu(self, event):
        """Mostra o menu de contexto."""
        # Seleciona o item sob o cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self._on_document_select(None)
            self.context_menu.post(event.x_root, event.y_root)
    
    def _on_edit_click(self):
        """Abre o dialog de edicao."""
        if not self.selected_doc_id:
            return
        
        # Busca dados atuais
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT 
                d.id, d.doc_type, d.entity_tag,
                t.amount_cents, t.supplier_clean, t.due_date, t.payment_date
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE d.id = ?
        """, (self.selected_doc_id,))
        
        row = cursor.fetchone()
        if not row:
            return
        
        doc_data = dict(row)
        
        # Abre dialog
        dialog = EditDialog(self.root, doc_data)
        self.root.wait_window(dialog)
        
        if dialog.result:
            # Atualiza no banco
            try:
                self.db.update_transaction_fields(
                    doc_id=self.selected_doc_id,
                    supplier_clean=dialog.result['supplier'],
                    amount_cents=dialog.result['amount_cents'],
                    due_date=dialog.result['due_date'],
                    payment_date=dialog.result['payment_date']
                )
                
                self._refresh_all()
                self._show_document_details(self.selected_doc_id)
                self._set_status("Dados atualizados com sucesso")
                
            except Exception as e:
                Messagebox.showerror(
                    title="Erro",
                    message=f"Erro ao salvar: {e}",
                    parent=self.root
                )
    
    def _on_reprocess_click(self):
        """Reprocessa o PDF selecionado."""
        if not self.selected_doc_id:
            return
        
        # Busca caminho original
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
        row = cursor.fetchone()
        if not row:
            return
        
        path = row['original_path']
        if not os.path.exists(path):
            Messagebox.showerror(
                title="Erro",
                message=f"Arquivo nao encontrado:\n{path}",
                parent=self.root
            )
            return
        
        # Confirma
        result = Messagebox.yesno(
            title="Reprocessar PDF",
            message="Isso ira excluir os dados atuais e reprocessar o arquivo.\n\nContinuar?",
            parent=self.root
        )
        
        if result == "Yes":
            # Remove do banco
            self._delete_document(self.selected_doc_id)
            
            # Reprocessa
            try:
                self.scanner.process_file(Path(path))
                self._refresh_all()
                self._set_status("PDF reprocessado com sucesso")
            except Exception as e:
                Messagebox.showerror(
                    title="Erro",
                    message=f"Erro ao reprocessar: {e}",
                    parent=self.root
                )
    
    def _on_open_file_click(self):
        """Abre o arquivo original no visualizador padrao."""
        if not self.selected_doc_id:
            return
        
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
        row = cursor.fetchone()
        if row and os.path.exists(row['original_path']):
            os.startfile(row['original_path'])
    
    def _on_open_folder_click(self):
        """Abre a pasta contendo o arquivo."""
        if not self.selected_doc_id:
            return
        
        cursor = self.db.connection.cursor()
        cursor.execute("SELECT original_path FROM documentos WHERE id = ?", (self.selected_doc_id,))
        row = cursor.fetchone()
        if row and os.path.exists(row['original_path']):
            folder = os.path.dirname(row['original_path'])
            os.startfile(folder)
    
    def _on_delete_click(self):
        """Exclui o documento do banco de dados."""
        if not self.selected_doc_id:
            return
        
        result = Messagebox.yesno(
            title="Confirmar Exclusao",
            message="Deseja excluir este documento do banco de dados?\n\n"
                    "O arquivo original NAO sera apagado.",
            parent=self.root
        )
        
        if result == "Yes":
            self._delete_document(self.selected_doc_id)
            self.selected_doc_id = None
            self._show_no_selection()
            self._refresh_all()
            self._set_status("Documento excluido do banco")
    
    def _delete_document(self, doc_id: int):
        """Remove um documento do banco de dados."""
        cursor = self.db.connection.cursor()
        
        # Remove matches relacionados
        cursor.execute("DELETE FROM matches WHERE parent_doc_id = ? OR child_doc_id = ?", (doc_id, doc_id))
        
        # Remove duplicatas
        cursor.execute("DELETE FROM duplicatas WHERE nfe_id = ? OR boleto_id = ?", (doc_id, doc_id))
        
        # Remove transacoes
        cursor.execute("DELETE FROM transacoes WHERE doc_id = ?", (doc_id,))
        
        # Remove documento
        cursor.execute("DELETE FROM documentos WHERE id = ?", (doc_id,))
        
        self.db.connection.commit()
    
    # ==========================================================================
    # DRAG & DROP
    # ==========================================================================
    
    def _setup_drag_drop(self):
        """Configura Drag & Drop de arquivos (desabilitado - requer TkinterDnD)."""
        # Nota: Drag & Drop nativo requer tkinterdnd2 que nao esta instalado
        # Por enquanto, use o botao '+ Adicionar PDFs' ou 'Escanear Pasta'
        pass
    
    # ==========================================================================
    # CONSTRUCAO DA INTERFACE
    # ==========================================================================
    
    def _build_ui(self):
        """Constroi todos os elementos da interface."""
        # Container principal
        self.main_container = ttk.Frame(self.root, padding=10)
        self.main_container.pack(fill=BOTH, expand=YES)
        
        # Cabecalho com botoes
        self._build_header()
        
        # Separador
        ttk.Separator(self.main_container, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        # Area de conteudo (duas colunas)
        self._build_content_area()
        
        # Barra de status
        self._build_status_bar()
    
    def _build_header(self):
        """Constroi o cabecalho com botoes de acao."""
        header = ttk.Frame(self.main_container)
        header.pack(fill=X, pady=(0, 5))
        
        # Titulo
        title_frame = ttk.Frame(header)
        title_frame.pack(side=LEFT)
        
        ttk.Label(
            title_frame,
            text="SRDA-Rural",
            font=("Helvetica", 20, "bold"),
            bootstyle="inverse-primary"
        ).pack(side=LEFT, padx=(0, 10))
        
        ttk.Label(
            title_frame,
            text="v2.0 - Sistema de Reconciliacao",
            font=("Helvetica", 10),
            bootstyle="secondary"
        ).pack(side=LEFT, pady=8)
        
        # Frame para botoes
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side=RIGHT)
        
        # Botao: Adicionar PDFs
        self.btn_add = ttk.Button(
            btn_frame,
            text="+ Adicionar PDFs",
            bootstyle="primary",
            width=16,
            command=self._on_add_files_click
        )
        self.btn_add.pack(side=LEFT, padx=5)
        ToolTip(self.btn_add, text="Adiciona arquivos PDF individuais")
        
        # Botao: Escanear Pasta
        self.btn_scan = ttk.Button(
            btn_frame,
            text="Escanear Pasta",
            bootstyle="info-outline",
            width=16,
            command=self._on_scan_click
        )
        self.btn_scan.pack(side=LEFT, padx=5)
        ToolTip(self.btn_scan, text="Escaneia uma pasta inteira de PDFs")
        
        # Botao: Processar Vinculos
        self.btn_match = ttk.Button(
            btn_frame,
            text="Processar Vinculos",
            bootstyle="success-outline",
            width=16,
            command=self._on_match_click
        )
        self.btn_match.pack(side=LEFT, padx=5)
        ToolTip(self.btn_match, text="Reconcilia documentos automaticamente")
        
        # Botao: Renomear Arquivos
        self.btn_rename = ttk.Button(
            btn_frame,
            text="Renomear Arquivos",
            bootstyle="warning-outline",
            width=16,
            command=self._on_rename_click
        )
        self.btn_rename.pack(side=LEFT, padx=5)
        ToolTip(self.btn_rename, text="Renomeia arquivos reconciliados")
        
        # Botao: Atualizar
        self.btn_refresh = ttk.Button(
            btn_frame,
            text="Atualizar",
            bootstyle="secondary",
            width=10,
            command=self._refresh_all
        )
        self.btn_refresh.pack(side=LEFT, padx=5)
    
    def _build_content_area(self):
        """Constroi a area de conteudo com duas colunas."""
        content = ttk.Frame(self.main_container)
        content.pack(fill=BOTH, expand=YES)
        
        # Configura grid - 55% esquerda, 45% direita
        content.columnconfigure(0, weight=55)
        content.columnconfigure(1, weight=45)
        content.rowconfigure(0, weight=1)
        
        # Painel Esquerdo: Lista de Documentos
        self._build_document_list(content)
        
        # Painel Direito: Detalhes + Preview
        self._build_details_panel(content)
    
    def _build_document_list(self, parent):
        """Constroi o painel esquerdo com a lista de documentos."""
        left_frame = ttk.Labelframe(
            parent,
            text=" Documentos Processados ",
            padding=10,
            bootstyle="primary"
        )
        left_frame.grid(row=0, column=0, sticky=NSEW, padx=(0, 5))
        
        # Frame para filtros
        filter_frame = ttk.Frame(left_frame)
        filter_frame.pack(fill=X, pady=(0, 10))
        
        # Filtro por entidade
        ttk.Label(filter_frame, text="Entidade:").pack(side=LEFT, padx=(0, 5))
        self.filter_entity = ttk.Combobox(
            filter_frame,
            values=["Todos", "VG - Vagner", "MV - Marcelli"],
            state="readonly",
            width=15
        )
        self.filter_entity.set("Todos")
        self.filter_entity.pack(side=LEFT, padx=(0, 15))
        self.filter_entity.bind("<<ComboboxSelected>>", lambda e: self._refresh_document_list())
        
        # Filtro por tipo
        ttk.Label(filter_frame, text="Tipo:").pack(side=LEFT, padx=(0, 5))
        self.filter_type = ttk.Combobox(
            filter_frame,
            values=["Todos", "NFE", "NFSE", "BOLETO", "COMPROVANTE"],
            state="readonly",
            width=15
        )
        self.filter_type.set("Todos")
        self.filter_type.pack(side=LEFT)
        self.filter_type.bind("<<ComboboxSelected>>", lambda e: self._refresh_document_list())
        
        # Filtro por status
        ttk.Label(filter_frame, text="Status:").pack(side=LEFT, padx=(15, 5))
        self.filter_status = ttk.Combobox(
            filter_frame,
            values=["Todos", "PARSED", "RECONCILED", "RENAMED"],
            state="readonly",
            width=12
        )
        self.filter_status.set("Todos")
        self.filter_status.pack(side=LEFT)
        self.filter_status.bind("<<ComboboxSelected>>", lambda e: self._refresh_document_list())
        
        # Treeview
        columns = ("id", "tipo", "entidade", "fornecedor", "valor", "status")
        self.tree = ttk.Treeview(
            left_frame,
            columns=columns,
            show="headings",
            bootstyle="primary",
            selectmode="browse"
        )
        
        # Configurar colunas
        self.tree.heading("id", text="ID")
        self.tree.heading("tipo", text="Tipo")
        self.tree.heading("entidade", text="Ent")
        self.tree.heading("fornecedor", text="Fornecedor")
        self.tree.heading("valor", text="Valor")
        self.tree.heading("status", text="Status")
        
        self.tree.column("id", width=40, anchor=CENTER)
        self.tree.column("tipo", width=70, anchor=CENTER)
        self.tree.column("entidade", width=40, anchor=CENTER)
        self.tree.column("fornecedor", width=180)
        self.tree.column("valor", width=100, anchor=E)
        self.tree.column("status", width=90, anchor=CENTER)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(left_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.tree.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Eventos
        self.tree.bind("<<TreeviewSelect>>", self._on_document_select)
        self.tree.bind("<Double-1>", lambda e: self._on_edit_click())  # Duplo clique = editar
    
    def _build_details_panel(self, parent):
        """Constroi o painel direito com detalhes e preview."""
        right_frame = ttk.Labelframe(
            parent,
            text=" Detalhes do Documento ",
            padding=10,
            bootstyle="info"
        )
        right_frame.grid(row=0, column=1, sticky=NSEW, padx=(5, 0))
        
        # Notebook com abas: Detalhes | Preview
        self.notebook = ttk.Notebook(right_frame, bootstyle="info")
        self.notebook.pack(fill=BOTH, expand=YES)
        
        # Aba: Detalhes
        self.details_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.details_tab, text=" Informacoes ")
        
        # Scrolled frame para detalhes
        self.details_scroll = ScrolledFrame(self.details_tab, autohide=True)
        self.details_scroll.pack(fill=BOTH, expand=YES)
        self.details_frame = self.details_scroll
        
        # Aba: Preview
        self.preview_tab = ttk.Frame(self.notebook, padding=5)
        self.notebook.add(self.preview_tab, text=" Preview PDF ")
        
        # Label para preview
        self.preview_label = ttk.Label(self.preview_tab, anchor=CENTER)
        self.preview_label.pack(fill=BOTH, expand=YES)
        
        # Botoes de acao rapida
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=X, pady=(10, 0))
        
        ttk.Button(
            action_frame,
            text="Editar",
            bootstyle="info-outline",
            command=self._on_edit_click,
            width=10
        ).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(
            action_frame,
            text="Abrir PDF",
            bootstyle="secondary-outline",
            command=self._on_open_file_click,
            width=10
        ).pack(side=LEFT, padx=(0, 5))
        
        ttk.Button(
            action_frame,
            text="Excluir",
            bootstyle="danger-outline",
            command=self._on_delete_click,
            width=10
        ).pack(side=RIGHT)
        
        # Placeholder inicial
        self._show_no_selection()
    
    def _build_status_bar(self):
        """Constroi a barra de status inferior."""
        status_bar = ttk.Frame(self.main_container)
        status_bar.pack(fill=X, pady=(10, 0))
        
        # Estatisticas
        self.stats_frame = ttk.Frame(status_bar)
        self.stats_frame.pack(side=LEFT)
        
        self.lbl_total = ttk.Label(
            self.stats_frame,
            text="Total: 0",
            bootstyle="inverse-secondary"
        )
        self.lbl_total.pack(side=LEFT, padx=(0, 15))
        
        self.lbl_reconciled = ttk.Label(
            self.stats_frame,
            text="Reconciliados: 0",
            bootstyle="inverse-success"
        )
        self.lbl_reconciled.pack(side=LEFT, padx=(0, 15))
        
        self.lbl_pending = ttk.Label(
            self.stats_frame,
            text="Pendentes: 0",
            bootstyle="inverse-warning"
        )
        self.lbl_pending.pack(side=LEFT, padx=(0, 15))
        
        # Dica
        ttk.Label(
            status_bar,
            text="Dica: Clique direito para mais opcoes",
            font=("Helvetica", 8),
            bootstyle="secondary"
        ).pack(side=LEFT, padx=20)
        
        # Status de operacao
        self.lbl_status = ttk.Label(
            status_bar,
            text="Pronto",
            bootstyle="secondary"
        )
        self.lbl_status.pack(side=RIGHT)
        
        # Progress bar (oculta por padrao)
        self.progress = ttk.Progressbar(
            status_bar,
            mode="indeterminate",
            bootstyle="success-striped"
        )
    
    # ==========================================================================
    # PREVIEW DO PDF
    # ==========================================================================
    
    def _show_pdf_preview(self, pdf_path: str, page_num: int = 0):
        """Exibe preview da primeira pagina do PDF."""
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                doc.close()
                return
            
            page = doc[min(page_num, len(doc) - 1)]
            
            # Calcula escala para caber no painel
            zoom = 0.8
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Converte para PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Redimensiona se necessario
            max_width = 400
            max_height = 500
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Converte para PhotoImage
            self.preview_image = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_image)
            
            doc.close()
            
        except Exception as e:
            self.preview_label.configure(image='', text=f"Erro ao carregar preview:\n{e}")
    
    # ==========================================================================
    # EXIBICAO DE DETALHES
    # ==========================================================================
    
    def _show_no_selection(self):
        """Mostra mensagem quando nenhum documento esta selecionado."""
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        frame = ttk.Frame(self.details_frame)
        frame.pack(expand=YES, fill=BOTH, padx=20, pady=50)
        
        ttk.Label(
            frame,
            text="Selecione um documento",
            font=("Helvetica", 12),
            bootstyle="secondary"
        ).pack()
        
        ttk.Label(
            frame,
            text="Clique em um item da lista para ver os detalhes",
            bootstyle="secondary"
        ).pack(pady=5)
        
        ttk.Label(
            frame,
            text="ou clique com botao direito para mais opcoes",
            bootstyle="secondary"
        ).pack(pady=2)
        
        # Limpa preview
        self.preview_label.configure(image='', text="Nenhum documento selecionado")
    
    def _show_document_details(self, doc_id: int):
        """Exibe os detalhes de um documento."""
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        # Busca dados do documento
        cursor = self.db.connection.cursor()
        cursor.execute("""
            SELECT 
                d.id, d.file_hash, d.original_path, d.doc_type, 
                d.entity_tag, d.status, d.created_at, d.page_start, d.page_end,
                d.access_key, d.doc_number,
                t.amount_cents, t.supplier_clean, t.emission_date, t.due_date,
                t.payment_date, t.is_scheduled
            FROM documentos d
            LEFT JOIN transacoes t ON d.id = t.doc_id
            WHERE d.id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if not row:
            self._show_no_selection()
            return
        
        doc = dict(row)
        
        # Container principal
        container = ttk.Frame(self.details_frame)
        container.pack(fill=BOTH, expand=YES, padx=10, pady=10)
        
        # Cabecalho
        header = ttk.Frame(container)
        header.pack(fill=X, pady=(0, 15))
        
        doc_type = doc.get('doc_type', 'UNKNOWN')
        icon = DOC_ICONS.get(doc_type, "[??]")
        
        ttk.Label(
            header,
            text=f"{icon} {doc_type}",
            font=("Helvetica", 16, "bold"),
            bootstyle="primary"
        ).pack(side=LEFT)
        
        entity = doc.get('entity_tag', '')
        if entity:
            entity_style = "success" if entity == "VG" else "info"
            ttk.Label(
                header,
                text=entity,
                font=("Helvetica", 14, "bold"),
                bootstyle=entity_style
            ).pack(side=RIGHT)
        
        # Separador
        ttk.Separator(container, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        # Informacoes principais
        info_frame = ttk.Frame(container)
        info_frame.pack(fill=X)
        
        row_num = 0
        
        def add_info_row(label: str, value: str, highlight: bool = False):
            nonlocal row_num
            ttk.Label(
                info_frame,
                text=label,
                font=("Helvetica", 10, "bold"),
                bootstyle="secondary"
            ).grid(row=row_num, column=0, sticky=W, pady=3)
            
            style = "success" if highlight else "default"
            ttk.Label(
                info_frame,
                text=value or "-",
                font=("Helvetica", 10),
                bootstyle=style,
                wraplength=250
            ).grid(row=row_num, column=1, sticky=W, padx=(10, 0), pady=3)
            row_num += 1
        
        # Dados
        add_info_row("ID:", str(doc.get('id', '')))
        add_info_row("Status:", doc.get('status', ''))
        
        # Valor
        amount = doc.get('amount_cents', 0)
        if amount:
            amount_display = f"R$ {SRDADatabase.cents_to_display(amount)}"
            add_info_row("Valor:", amount_display, highlight=True)
        
        # Fornecedor
        supplier = doc.get('supplier_clean', '')
        add_info_row("Fornecedor:", supplier)
        
        # Datas
        payment = doc.get('payment_date', '')
        due = doc.get('due_date', '')
        emission = doc.get('emission_date', '')
        
        if payment:
            add_info_row("Data Pagamento:", payment, highlight=True)
        if due:
            add_info_row("Data Vencimento:", due)
        if emission:
            add_info_row("Data Emissao:", emission)
        
        # Numero
        doc_num = doc.get('doc_number', '')
        if doc_num:
            add_info_row("Numero:", doc_num)
        
        # Paginas
        page_start = doc.get('page_start', 1)
        page_end = doc.get('page_end', 1)
        if page_start and page_end and page_start != page_end:
            add_info_row("Paginas:", f"{page_start} a {page_end}")
        
        # Separador
        ttk.Separator(container, orient=HORIZONTAL).pack(fill=X, pady=15)
        
        # Arquivo original
        ttk.Label(
            container,
            text="Arquivo Original:",
            font=("Helvetica", 10, "bold"),
            bootstyle="secondary"
        ).pack(anchor=W)
        
        path = doc.get('original_path', '')
        ttk.Label(
            container,
            text=Path(path).name if path else "-",
            font=("Helvetica", 9),
            bootstyle="info",
            wraplength=300
        ).pack(anchor=W, pady=(2, 10))
        
        # Mostra preview
        if path and os.path.exists(path):
            page_idx = (doc.get('page_start', 1) or 1) - 1
            self._show_pdf_preview(path, page_idx)
        
        # Vinculos
        self._show_document_matches(container, doc_id)
    
    def _show_document_matches(self, parent, doc_id: int):
        """Exibe os vinculos de um documento."""
        matches = self.db.get_matches_by_document(doc_id)
        
        if not matches:
            return
        
        ttk.Separator(parent, orient=HORIZONTAL).pack(fill=X, pady=10)
        
        ttk.Label(
            parent,
            text="Vinculos:",
            font=("Helvetica", 10, "bold"),
            bootstyle="secondary"
        ).pack(anchor=W)
        
        for match in matches:
            match_frame = ttk.Frame(parent)
            match_frame.pack(fill=X, pady=5)
            
            if match['parent_doc_id'] == doc_id:
                direction = "→"
                other_path = match['child_path']
            else:
                direction = "←"
                other_path = match['parent_path']
            
            confidence = match.get('confidence', 0) * 100
            
            ttk.Label(
                match_frame,
                text=f"{direction} {Path(other_path).name}",
                font=("Helvetica", 9),
                bootstyle="info"
            ).pack(side=LEFT)
            
            ttk.Label(
                match_frame,
                text=f"({confidence:.0f}%)",
                font=("Helvetica", 9),
                bootstyle="success" if confidence >= 70 else "warning"
            ).pack(side=RIGHT)
    
    # ==========================================================================
    # ATUALIZACAO DE DADOS
    # ==========================================================================
    
    def _refresh_document_list(self):
        """Atualiza a lista de documentos."""
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
        
        # Filtros
        entity_filter = self.filter_entity.get()
        if entity_filter == "VG - Vagner":
            query += " AND d.entity_tag = 'VG'"
        elif entity_filter == "MV - Marcelli":
            query += " AND d.entity_tag = 'MV'"
        
        type_filter = self.filter_type.get()
        if type_filter != "Todos":
            query += f" AND d.doc_type = '{type_filter}'"
        
        status_filter = self.filter_status.get()
        if status_filter != "Todos":
            query += f" AND d.status = '{status_filter}'"
        
        query += " ORDER BY d.id DESC"
        
        cursor.execute(query)
        
        for row in cursor.fetchall():
            doc = dict(row)
            
            doc_type = doc.get('doc_type', 'UNKNOWN')
            entity = doc.get('entity_tag', '-') or '-'
            supplier = doc.get('supplier_clean', '-') or '-'
            status = doc.get('status', '-')
            
            amount = doc.get('amount_cents', 0)
            amount_str = f"R$ {SRDADatabase.cents_to_display(amount)}" if amount else "-"
            
            if len(supplier) > 22:
                supplier = supplier[:19] + "..."
            
            self.tree.insert(
                "",
                END,
                values=(doc['id'], doc_type, entity, supplier, amount_str, status),
                tags=(status,)
            )
        
        for status, style in STATUS_COLORS.items():
            self.tree.tag_configure(status, foreground=self._get_bootstrap_color(style))
    
    def _get_bootstrap_color(self, style: str) -> str:
        """Retorna a cor hex para um estilo bootstrap."""
        colors = {
            "primary": "#0d6efd",
            "secondary": "#6c757d",
            "success": "#198754",
            "danger": "#dc3545",
            "warning": "#ffc107",
            "info": "#0dcaf0",
        }
        return colors.get(style, "#ffffff")
    
    def _update_statistics(self):
        """Atualiza as estatisticas na barra de status."""
        stats = self.db.get_statistics()
        
        total = stats.get('total_documents', 0)
        by_status = stats.get('by_status', {})
        
        reconciled = by_status.get('RECONCILED', 0) + by_status.get('RENAMED', 0)
        pending = total - reconciled
        
        self.lbl_total.configure(text=f"Total: {total}")
        self.lbl_reconciled.configure(text=f"Reconciliados: {reconciled}")
        self.lbl_pending.configure(text=f"Pendentes: {pending}")
    
    def _refresh_all(self):
        """Atualiza todos os dados."""
        self._refresh_document_list()
        self._update_statistics()
        self._set_status("Lista atualizada")
    
    # ==========================================================================
    # EVENTOS
    # ==========================================================================
    
    def _on_document_select(self, event):
        """Evento de selecao de documento."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        doc_id = item['values'][0]
        
        self.selected_doc_id = doc_id
        self._show_document_details(doc_id)
    
    def _on_add_files_click(self):
        """Adiciona arquivos PDF individuais."""
        files = filedialog.askopenfilenames(
            title="Selecione os arquivos PDF",
            filetypes=[("Arquivos PDF", "*.pdf"), ("Todos", "*.*")]
        )
        
        if files:
            for f in files:
                try:
                    self.scanner.process_file(Path(f))
                except Exception as e:
                    print(f"Erro ao processar {f}: {e}")
            
            self._refresh_all()
            self._set_status(f"{len(files)} arquivo(s) adicionado(s)")
    
    def _on_scan_click(self):
        """Evento de clique no botao Escanear."""
        if self.is_processing:
            return
        
        folder = filedialog.askdirectory(
            title="Selecione a pasta com os PDFs",
            initialdir="."
        )
        
        if folder:
            self._run_in_thread(self._do_scan, folder)
    
    def _on_match_click(self):
        """Evento de clique no botao Processar Vinculos."""
        if self.is_processing:
            return
        
        self._run_in_thread(self._do_match)
    
    def _on_rename_click(self):
        """Evento de clique no botao Renomear."""
        if self.is_processing:
            return
        
        result = Messagebox.yesno(
            title="Confirmar Renomeacao",
            message="Deseja renomear os arquivos reconciliados?\n\n"
                    "Os arquivos serao COPIADOS para a pasta 'Output'.\n"
                    "Os originais nao serao alterados.",
            parent=self.root
        )
        
        if result == "Yes":
            self._run_in_thread(self._do_rename)
    
    # ==========================================================================
    # OPERACOES EM THREAD
    # ==========================================================================
    
    def _run_in_thread(self, func, *args):
        """Executa uma funcao em thread separada."""
        self.is_processing = True
        self._set_buttons_state(False)
        self._show_progress(True)
        
        def wrapper():
            try:
                func(*args)
            except Exception as e:
                self.root.after(0, lambda: Messagebox.showerror(
                    title="Erro",
                    message=str(e),
                    parent=self.root
                ))
            finally:
                self.root.after(0, self._on_operation_complete)
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
    
    def _on_operation_complete(self):
        """Chamado quando uma operacao termina."""
        self.is_processing = False
        self._set_buttons_state(True)
        self._show_progress(False)
        self._refresh_all()
    
    def _set_buttons_state(self, enabled: bool):
        """Habilita/desabilita botoes."""
        state = NORMAL if enabled else DISABLED
        self.btn_add.configure(state=state)
        self.btn_scan.configure(state=state)
        self.btn_match.configure(state=state)
        self.btn_rename.configure(state=state)
    
    def _show_progress(self, show: bool):
        """Mostra/oculta a barra de progresso."""
        if show:
            self.progress.pack(side=RIGHT, padx=(10, 0))
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.pack_forget()
    
    def _set_status(self, text: str):
        """Atualiza o texto de status."""
        self.lbl_status.configure(text=text)
    
    # ==========================================================================
    # OPERACOES
    # ==========================================================================
    
    def _do_scan(self, folder: str):
        """Executa o escaneamento."""
        self.root.after(0, lambda: self._set_status("Escaneando..."))
        
        self.scanner.input_folder = Path(folder)
        stats = self.scanner.process_all()
        
        msg = f"Processados: {stats['processed']} | Criados: {stats['documents_created']}"
        self.root.after(0, lambda: self._set_status(msg))
    
    def _do_match(self):
        """Executa a reconciliacao."""
        self.root.after(0, lambda: self._set_status("Processando vinculos..."))
        
        result = self.matcher.reconcile_all(auto_confirm=True)
        
        msg = f"Vinculados: {result.matched_boletos} | Pendentes: {result.unmatched_boletos}"
        self.root.after(0, lambda: self._set_status(msg))
    
    def _do_rename(self):
        """Executa a renomeacao."""
        self.root.after(0, lambda: self._set_status("Renomeando arquivos..."))
        
        result = self.renamer.run(dry_run=False, copy_mode=True)
        
        msg = f"Renomeados: {result.successful} | Falhas: {result.failed}"
        self.root.after(0, lambda: self._set_status(msg))
    
    # ==========================================================================
    # EXECUCAO
    # ==========================================================================
    
    def run(self):
        """Inicia a aplicacao."""
        self.root.mainloop()
    
    def cleanup(self):
        """Limpa recursos ao fechar."""
        self.db.close()


# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

def main():
    """Funcao principal."""
    app = SRDAApplication()
    
    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
