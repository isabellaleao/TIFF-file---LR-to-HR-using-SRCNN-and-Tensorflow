#!/usr/bin/env python3
"""
Interface gráfica para visualização de mapas de super-resolução.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from universal_hybrid_pipeline import universal_hybrid_super_resolution

class MapVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("🌍 Visualizador de Super-Resolução")
        self.root.geometry("1400x900")
        
        # Dados carregados
        self.original_data = None
        self.result_data = None
        self.original_path = None
        self.result_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface do usuário."""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="🌍 Visualizador de Super-Resolução", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame de controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Primeira linha de botões
        button_frame1 = ttk.Frame(control_frame)
        button_frame1.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(button_frame1, text="📁 Carregar Mapa Original", 
                  command=self.load_original_map).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(button_frame1, text="📁 Carregar Resultado", 
                  command=self.load_result_map).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(button_frame1, text="🚀 Gerar Super-Resolução", 
                  command=self.generate_super_resolution).grid(row=0, column=2, padx=(0, 10))
        
        ttk.Button(button_frame1, text="💾 Salvar Comparação", 
                  command=self.save_comparison).grid(row=0, column=3)
        
        # Segunda linha de botões
        button_frame2 = ttk.Frame(control_frame)
        button_frame2.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame2, text="🔧 Pipeline Universal", 
                  command=self.run_universal_pipeline).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(button_frame2, text="🎯 Pipeline Híbrido", 
                  command=self.run_hybrid_pipeline).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(button_frame2, text="🧠 Treinar Modelo", 
                  command=self.run_training).grid(row=0, column=2, padx=(0, 10))
        
        ttk.Button(button_frame2, text="📊 Diagnóstico", 
                  command=self.run_diagnosis).grid(row=0, column=3)
        
        # Terceira linha de botões
        button_frame3 = ttk.Frame(control_frame)
        button_frame3.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(button_frame3, text="🧹 Limpar Tudo", 
                  command=self.clear_all_maps).grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(button_frame3, text="🔄 Recarregar Interface", 
                  command=self.reload_interface).grid(row=0, column=1)
        
        ttk.Button(button_frame3, text="📊 Carregar Ambos", 
                  command=self.load_both_maps).grid(row=0, column=2, padx=(10, 0))
        
        # Informações dos arquivos
        info_frame = ttk.LabelFrame(main_frame, text="Informações", padding="10")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=4, width=80, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar para info
        info_scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        # Frame dos gráficos
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.columnconfigure(1, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Gráfico original
        self.original_frame = ttk.LabelFrame(plot_frame, text="Mapa Original", padding="5")
        self.original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        self.original_frame.columnconfigure(0, weight=1)
        self.original_frame.rowconfigure(0, weight=1)
        
        # Gráfico resultado
        self.result_frame = ttk.LabelFrame(plot_frame, text="Super-Resolução", padding="5")
        self.result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)
        
        # Inicializar gráficos vazios
        self.setup_empty_plots()
        
    def setup_empty_plots(self):
        """Configura gráficos vazios iniciais."""
        
        # Gráfico original vazio
        self.original_fig = Figure(figsize=(6, 4), dpi=100)
        self.original_ax = self.original_fig.add_subplot(111)
        self.original_ax.set_title("Nenhum mapa carregado")
        self.original_ax.text(0.5, 0.5, "Clique em 'Carregar Mapa Original'", 
                             ha='center', va='center', transform=self.original_ax.transAxes)
        
        self.original_canvas = FigureCanvasTkAgg(self.original_fig, self.original_frame)
        self.original_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Gráfico resultado vazio
        self.result_fig = Figure(figsize=(6, 4), dpi=100)
        self.result_ax = self.result_fig.add_subplot(111)
        self.result_ax.set_title("Nenhum resultado carregado")
        self.result_ax.text(0.5, 0.5, "Clique em 'Gerar Super-Resolução'", 
                           ha='center', va='center', transform=self.result_ax.transAxes)
        
        self.result_canvas = FigureCanvasTkAgg(self.result_fig, self.result_frame)
        self.result_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def load_original_map(self):
        """Carrega mapa original."""
        
        file_path = filedialog.askopenfilename(
            title="Selecionar Mapa Original",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Limpar apenas o gráfico original (manter o resultado se existir)
            self.clear_original_plot()
            
            # Carregar novos dados
            self.original_data = self.load_tiff_data(file_path)
            self.original_path = file_path
            self.plot_original_map()
            self.update_info()
            messagebox.showinfo("Sucesso", f"Mapa original carregado: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar mapa: {str(e)}")
    
    def load_result_map(self):
        """Carrega mapa de resultado."""
        
        file_path = filedialog.askopenfilename(
            title="Selecionar Resultado de Super-Resolução",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Limpar apenas o gráfico de resultado (manter o original se existir)
            self.clear_result_plot()
            
            # Carregar novos dados
            self.result_data = self.load_tiff_data(file_path)
            self.result_path = file_path
            self.plot_result_map()
            self.update_info()
            messagebox.showinfo("Sucesso", f"Resultado carregado: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar resultado: {str(e)}")
    
    def load_both_maps(self):
        """Carrega ambos os mapas (original e resultado) simultaneamente."""
        
        # Selecionar mapa original
        original_path = filedialog.askopenfilename(
            title="Selecionar Mapa Original",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not original_path:
            return
        
        # Selecionar mapa de resultado
        result_path = filedialog.askopenfilename(
            title="Selecionar Resultado de Super-Resolução",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not result_path:
            return
        
        try:
            # Limpar tudo primeiro
            self.clear_all_maps()
            
            # Carregar mapa original
            self.original_data = self.load_tiff_data(original_path)
            self.original_path = original_path
            self.plot_original_map()
            
            # Carregar mapa de resultado
            self.result_data = self.load_tiff_data(result_path)
            self.result_path = result_path
            self.plot_result_map()
            
            # Atualizar informações
            self.update_info()
            
            messagebox.showinfo("Sucesso", 
                f"Ambos os mapas carregados:\n"
                f"Original: {os.path.basename(original_path)}\n"
                f"Resultado: {os.path.basename(result_path)}")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar mapas: {str(e)}")
    
    def load_tiff_data(self, file_path):
        """Carrega dados de arquivo TIFF."""
        
        ds = gdal.Open(file_path)
        if ds is None:
            raise RuntimeError(f"Não foi possível abrir: {file_path}")
        
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        
        # Filtrar NoData
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        
        ds = None
        return data
    
    def generate_super_resolution(self):
        """Gera super-resolução usando o pipeline universal."""
        
        if self.original_path is None:
            messagebox.showwarning("Aviso", "Carregue um mapa original primeiro!")
            return
        
        # Diálogo de configuração
        config_window = self.create_config_dialog()
        self.root.wait_window(config_window)
    
    def create_config_dialog(self):
        """Cria diálogo de configuração para super-resolução."""
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuração de Super-Resolução")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Centralizar diálogo
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Variáveis
        scale_factor = tk.IntVar(value=3)
        patch_size = tk.IntVar(value=30)
        overlap = tk.IntVar(value=10)
        min_valid_ratio = tk.DoubleVar(value=0.1)
        
        # Frame principal
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(main_frame, text="Configurações de Super-Resolução", 
                 font=("Arial", 12, "bold")).pack(pady=(0, 20))
        
        # Escala
        ttk.Label(main_frame, text="Fator de Escala:").pack(anchor=tk.W)
        scale_frame = ttk.Frame(main_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Scale(scale_frame, from_=2, to=5, variable=scale_factor, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(scale_frame, textvariable=scale_factor).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Tamanho do patch
        ttk.Label(main_frame, text="Tamanho do Patch:").pack(anchor=tk.W)
        patch_frame = ttk.Frame(main_frame)
        patch_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Scale(patch_frame, from_=20, to=50, variable=patch_size, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(patch_frame, textvariable=patch_size).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Overlap
        ttk.Label(main_frame, text="Sobreposição:").pack(anchor=tk.W)
        overlap_frame = ttk.Frame(main_frame)
        overlap_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Scale(overlap_frame, from_=5, to=20, variable=overlap, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(overlap_frame, textvariable=overlap).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Proporção mínima válida
        ttk.Label(main_frame, text="Proporção Mínima Válida:").pack(anchor=tk.W)
        valid_frame = ttk.Frame(main_frame)
        valid_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Scale(valid_frame, from_=0.05, to=0.5, variable=min_valid_ratio, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(valid_frame, textvariable=min_valid_ratio).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Botões
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def start_processing():
            dialog.destroy()
            self.run_super_resolution(
                scale_factor.get(), patch_size.get(), 
                overlap.get(), min_valid_ratio.get()
            )
        
        ttk.Button(button_frame, text="🚀 Iniciar", command=start_processing).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="❌ Cancelar", command=dialog.destroy).pack(side=tk.LEFT)
        
        return dialog
    
    def run_super_resolution(self, scale_factor, patch_size, overlap, min_valid_ratio):
        """Executa super-resolução com progresso."""
        
        # Criar janela de progresso
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processando Super-Resolução")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Centralizar
        progress_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
        
        # Frame de progresso
        progress_frame = ttk.Frame(progress_window, padding="20")
        progress_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(progress_frame, text="Processando super-resolução...", 
                 font=("Arial", 12)).pack(pady=(0, 20))
        
        progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        progress_bar.pack(fill=tk.X, pady=(0, 20))
        progress_bar.start()
        
        ttk.Label(progress_frame, text="Isso pode levar alguns minutos...").pack()
        
        # Atualizar interface
        self.root.update()
        
        try:
            # Gerar nome de saída
            base_name = os.path.splitext(os.path.basename(self.original_path))[0]
            output_path = f"geospatial_output/{base_name}_super_resolution_{scale_factor}x.tif"
            
            # Executar super-resolução
            universal_hybrid_super_resolution(
                input_path=self.original_path,
                model_path='geospatial_output/advanced_model_best.keras',
                output_path=output_path,
                scale_factor=scale_factor,
                patch_size=patch_size,
                overlap=overlap,
                min_valid_ratio=min_valid_ratio
            )
            
            # Carregar resultado
            self.result_data = self.load_tiff_data(output_path)
            self.result_path = output_path
            self.plot_result_map()
            self.update_info()
            
            progress_window.destroy()
            messagebox.showinfo("Sucesso", f"Super-resolução concluída!\nSalvo em: {output_path}")
            
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Erro", f"Erro durante processamento: {str(e)}")
    
    def plot_original_map(self):
        """Plota mapa original com sombreamento."""
        
        if self.original_data is None:
            return
        
        # Limpar completamente o gráfico e figura
        self.original_ax.clear()
        
        # Remover todas as colorbars existentes
        for child in list(self.original_fig.get_children()):
            if hasattr(child, 'remove') and 'colorbar' in str(type(child)).lower():
                child.remove()
        
        # Filtrar NaN para visualização
        data_clean = np.where(np.isnan(self.original_data), 0, self.original_data)
        
        # Criar sombreamento (shaded relief)
        shaded_data = self.create_shaded_relief(data_clean)
        
        # Plotar imagem
        im = self.original_ax.imshow(shaded_data, aspect='equal')
        self.original_ax.set_title(f"Original (Sombreamento)\n{self.original_data.shape[1]}x{self.original_data.shape[0]}")
        self.original_ax.set_xlabel("X")
        self.original_ax.set_ylabel("Y")
        
        # Adicionar nova colorbar
        self.original_fig.colorbar(im, ax=self.original_ax, shrink=0.8)
        
        # Redesenhar
        self.original_canvas.draw()
    
    def plot_result_map(self):
        """Plota mapa de resultado com sombreamento."""
        
        if self.result_data is None:
            return
        
        # Limpar completamente o gráfico e figura
        self.result_ax.clear()
        
        # Remover todas as colorbars existentes
        for child in list(self.result_fig.get_children()):
            if hasattr(child, 'remove') and 'colorbar' in str(type(child)).lower():
                child.remove()
        
        # Filtrar NaN para visualização
        data_clean = np.where(np.isnan(self.result_data), 0, self.result_data)
        
        # Criar sombreamento (shaded relief)
        shaded_data = self.create_shaded_relief(data_clean)
        
        # Plotar imagem
        im = self.result_ax.imshow(shaded_data, aspect='equal')
        self.result_ax.set_title(f"Super-Resolução (Sombreamento)\n{self.result_data.shape[1]}x{self.result_data.shape[0]}")
        self.result_ax.set_xlabel("X")
        self.result_ax.set_ylabel("Y")
        
        # Adicionar nova colorbar
        self.result_fig.colorbar(im, ax=self.result_ax, shrink=0.8)
        
        # Redesenhar
        self.result_canvas.draw()
    
    def save_comparison(self):
        """Salva comparação lado a lado."""
        
        if self.original_data is None or self.result_data is None:
            messagebox.showwarning("Aviso", "Carregue ambos os mapas primeiro!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Salvar Comparação",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Criar figura de comparação
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original com sombreamento
            data_clean_orig = np.where(np.isnan(self.original_data), 0, self.original_data)
            shaded_orig = self.create_shaded_relief(data_clean_orig)
            im1 = ax1.imshow(shaded_orig, aspect='equal')
            ax1.set_title(f"Original (Sombreamento)\n{self.original_data.shape[1]}x{self.original_data.shape[0]}")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            
            # Resultado com sombreamento
            data_clean_result = np.where(np.isnan(self.result_data), 0, self.result_data)
            shaded_result = self.create_shaded_relief(data_clean_result)
            im2 = ax2.imshow(shaded_result, aspect='equal')
            ax2.set_title(f"Super-Resolução (Sombreamento)\n{self.result_data.shape[1]}x{self.result_data.shape[0]}")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            messagebox.showinfo("Sucesso", f"Comparação salva em: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar comparação: {str(e)}")
    
    def update_info(self):
        """Atualiza informações dos arquivos."""
        
        info_text = "📊 Informações dos Arquivos:\n\n"
        
        if self.original_data is not None:
            info_text += f"🗺️  Original: {os.path.basename(self.original_path)}\n"
            info_text += f"   Dimensões: {self.original_data.shape[1]}x{self.original_data.shape[0]}\n"
            info_text += f"   Valores: {np.nanmin(self.original_data):.2f} a {np.nanmax(self.original_data):.2f}\n"
            info_text += f"   Dados válidos: {np.sum(~np.isnan(self.original_data))/self.original_data.size*100:.1f}%\n\n"
        
        if self.result_data is not None:
            info_text += f"🎯 Super-Resolução: {os.path.basename(self.result_path)}\n"
            info_text += f"   Dimensões: {self.result_data.shape[1]}x{self.result_data.shape[0]}\n"
            info_text += f"   Valores: {np.nanmin(self.result_data):.2f} a {np.nanmax(self.result_data):.2f}\n"
            info_text += f"   Dados válidos: {np.sum(~np.isnan(self.result_data))/self.result_data.size*100:.1f}%\n"
            
            if self.original_data is not None:
                scale_x = self.result_data.shape[1] / self.original_data.shape[1]
                scale_y = self.result_data.shape[0] / self.original_data.shape[0]
                info_text += f"   Escala: {scale_x:.1f}x (X) x {scale_y:.1f}x (Y)\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def run_universal_pipeline(self):
        """Executa pipeline universal."""
        
        if self.original_path is None:
            messagebox.showwarning("Aviso", "Carregue um mapa original primeiro!")
            return
        
        try:
            # Executar pipeline universal
            universal_hybrid_super_resolution(
                input_path=self.original_path,
                output_path='geospatial_output/universal_result.tif'
            )
            
            # Carregar resultado
            self.result_data = self.load_tiff_data('geospatial_output/universal_result.tif')
            self.result_path = 'geospatial_output/universal_result.tif'
            self.plot_result_map()
            self.update_info()
            
            messagebox.showinfo("Sucesso", "Pipeline universal executado com sucesso!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro no pipeline universal: {str(e)}")
    
    def run_hybrid_pipeline(self):
        """Executa pipeline híbrido."""
        
        if self.original_path is None:
            messagebox.showwarning("Aviso", "Carregue um mapa original primeiro!")
            return
        
        try:
            # Importar e executar pipeline híbrido
            from hybrid_model_inference import hybrid_super_resolution
            
            hybrid_super_resolution(
                anadem_path=self.original_path,
                output_path='geospatial_output/hybrid_result.tif'
            )
            
            # Carregar resultado
            self.result_data = self.load_tiff_data('geospatial_output/hybrid_result.tif')
            self.result_path = 'geospatial_output/hybrid_result.tif'
            self.plot_result_map()
            self.update_info()
            
            messagebox.showinfo("Sucesso", "Pipeline híbrido executado com sucesso!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro no pipeline híbrido: {str(e)}")
    
    def run_training(self):
        """Executa treinamento do modelo."""
        
        try:
            # Criar janela de progresso
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Treinando Modelo")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            # Centralizar
            progress_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 100, self.root.winfo_rooty() + 100))
            
            # Frame de progresso
            progress_frame = ttk.Frame(progress_window, padding="20")
            progress_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(progress_frame, text="Treinando modelo...", 
                     font=("Arial", 12)).pack(pady=(0, 20))
            
            progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
            progress_bar.pack(fill=tk.X, pady=(0, 20))
            progress_bar.start()
            
            ttk.Label(progress_frame, text="Isso pode levar alguns minutos...").pack()
            
            # Atualizar interface
            self.root.update()
            
            # Executar treinamento
            from improved_training import main as train_main
            train_main()
            
            progress_window.destroy()
            messagebox.showinfo("Sucesso", "Treinamento concluído com sucesso!")
            
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("Erro", f"Erro durante treinamento: {str(e)}")
    
    def run_diagnosis(self):
        """Executa diagnóstico dos dados."""
        
        if self.original_data is None:
            messagebox.showwarning("Aviso", "Carregue um mapa original primeiro!")
            return
        
        try:
            # Criar janela de diagnóstico
            diag_window = tk.Toplevel(self.root)
            diag_window.title("Diagnóstico dos Dados")
            diag_window.geometry("600x400")
            diag_window.transient(self.root)
            
            # Frame principal
            main_frame = ttk.Frame(diag_window, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Título
            ttk.Label(main_frame, text="📊 Diagnóstico dos Dados", 
                     font=("Arial", 14, "bold")).pack(pady=(0, 20))
            
            # Área de texto para resultados
            text_frame = ttk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Courier", 10))
            scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Executar diagnóstico
            diag_text = self.perform_diagnosis()
            text_widget.insert(tk.END, diag_text)
            text_widget.config(state=tk.DISABLED)
            
            # Botão fechar
            ttk.Button(main_frame, text="Fechar", command=diag_window.destroy).pack(pady=(20, 0))
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante diagnóstico: {str(e)}")
    
    def perform_diagnosis(self):
        """Executa diagnóstico dos dados carregados."""
        
        diag_text = "🔍 DIAGNÓSTICO DOS DADOS\n"
        diag_text += "=" * 50 + "\n\n"
        
        if self.original_data is not None:
            diag_text += "🗺️  MAPA ORIGINAL:\n"
            diag_text += f"   Arquivo: {os.path.basename(self.original_path)}\n"
            diag_text += f"   Dimensões: {self.original_data.shape[1]} x {self.original_data.shape[0]}\n"
            diag_text += f"   Tipo de dados: {self.original_data.dtype}\n"
            
            # Estatísticas
            valid_data = self.original_data[~np.isnan(self.original_data)]
            if len(valid_data) > 0:
                diag_text += f"   Valores válidos: {len(valid_data):,} ({len(valid_data)/self.original_data.size*100:.1f}%)\n"
                diag_text += f"   Mínimo: {np.min(valid_data):.2f}\n"
                diag_text += f"   Máximo: {np.max(valid_data):.2f}\n"
                diag_text += f"   Média: {np.mean(valid_data):.2f}\n"
                diag_text += f"   Desvio padrão: {np.std(valid_data):.2f}\n"
                diag_text += f"   Mediana: {np.median(valid_data):.2f}\n"
            else:
                diag_text += "   ⚠️  Nenhum dado válido encontrado!\n"
            
            diag_text += "\n"
        
        if self.result_data is not None:
            diag_text += "🎯 SUPER-RESOLUÇÃO:\n"
            diag_text += f"   Arquivo: {os.path.basename(self.result_path)}\n"
            diag_text += f"   Dimensões: {self.result_data.shape[1]} x {self.result_data.shape[0]}\n"
            diag_text += f"   Tipo de dados: {self.result_data.dtype}\n"
            
            # Estatísticas
            valid_data = self.result_data[~np.isnan(self.result_data)]
            if len(valid_data) > 0:
                diag_text += f"   Valores válidos: {len(valid_data):,} ({len(valid_data)/self.result_data.size*100:.1f}%)\n"
                diag_text += f"   Mínimo: {np.min(valid_data):.2f}\n"
                diag_text += f"   Máximo: {np.max(valid_data):.2f}\n"
                diag_text += f"   Média: {np.mean(valid_data):.2f}\n"
                diag_text += f"   Desvio padrão: {np.std(valid_data):.2f}\n"
                diag_text += f"   Mediana: {np.median(valid_data):.2f}\n"
                
                # Comparação com original
                if self.original_data is not None:
                    scale_x = self.result_data.shape[1] / self.original_data.shape[1]
                    scale_y = self.result_data.shape[0] / self.original_data.shape[0]
                    diag_text += f"   Escala: {scale_x:.1f}x (X) x {scale_y:.1f}x (Y)\n"
                    
                    # Verificar se há melhoria na resolução
                    if scale_x > 1 and scale_y > 1:
                        diag_text += f"   ✅ Resolução aumentada em {scale_x:.1f}x\n"
                    else:
                        diag_text += f"   ⚠️  Resolução não aumentada\n"
            else:
                diag_text += "   ⚠️  Nenhum dado válido encontrado!\n"
            
            diag_text += "\n"
        
        # Recomendações
        diag_text += "💡 RECOMENDAÇÕES:\n"
        if self.original_data is None:
            diag_text += "   • Carregue um mapa original para começar\n"
        elif self.result_data is None:
            diag_text += "   • Execute super-resolução para gerar resultado\n"
        else:
            diag_text += "   • Compare os mapas visualmente\n"
            diag_text += "   • Verifique se a resolução foi aumentada\n"
            diag_text += "   • Salve a comparação se estiver satisfeito\n"
        
        return diag_text
    
    def clear_all_maps(self):
        """Limpa todos os mapas e gráficos."""
        
        # Limpar dados
        self.original_data = None
        self.result_data = None
        self.original_path = None
        self.result_path = None
        
        # Limpeza mais agressiva das figuras
        self.force_clear_figures()
        
        # Atualizar informações
        self.update_info()
        
        print("🧹 Todos os mapas foram limpos")
    
    def force_clear_figures(self):
        """Força a limpeza completa das figuras."""
        
        try:
            # Limpar figura original
            self.original_fig.clear()
            self.original_ax = self.original_fig.add_subplot(111)
            self.original_ax.set_title("Nenhum mapa carregado")
            self.original_ax.text(0.5, 0.5, "Clique em 'Carregar Mapa Original'", 
                                 ha='center', va='center', transform=self.original_ax.transAxes)
            self.original_canvas.draw()
            
            # Limpar figura de resultado
            self.result_fig.clear()
            self.result_ax = self.result_fig.add_subplot(111)
            self.result_ax.set_title("Nenhum resultado carregado")
            self.result_ax.text(0.5, 0.5, "Clique em 'Gerar Super-Resolução'", 
                               ha='center', va='center', transform=self.result_ax.transAxes)
            self.result_canvas.draw()
            
        except Exception as e:
            print(f"Erro ao forçar limpeza: {e}")
            # Fallback para limpeza normal
            self.clear_original_plot()
            self.clear_result_plot()
    
    def reload_interface(self):
        """Recarrega completamente a interface."""
        
        # Limpar tudo
        self.clear_all_maps()
        
        # Recriar gráficos vazios
        self.setup_empty_plots()
        
        # Atualizar informações
        self.update_info()
        
        messagebox.showinfo("Sucesso", "Interface recarregada com sucesso!")
    
    def clear_original_plot(self):
        """Limpa completamente o gráfico original."""
        
        try:
            # Limpar axes
            self.original_ax.clear()
            
            # Remover TODAS as colorbars de forma mais agressiva
            for child in list(self.original_fig.get_children()):
                try:
                    if hasattr(child, 'remove'):
                        child_type = str(type(child)).lower()
                        if 'colorbar' in child_type or 'colorbar' in str(child):
                            child.remove()
                except:
                    pass  # Ignorar erros de remoção
            
            # Remover também do axes
            for child in list(self.original_ax.get_children()):
                try:
                    if hasattr(child, 'remove'):
                        child_type = str(type(child)).lower()
                        if 'colorbar' in child_type or 'colorbar' in str(child):
                            child.remove()
                except:
                    pass  # Ignorar erros de remoção
            
            # Resetar limites dos eixos
            self.original_ax.set_xlim(0, 1)
            self.original_ax.set_ylim(0, 1)
            
            # Remover eixos desnecessários
            self.original_ax.set_xticks([])
            self.original_ax.set_yticks([])
            
            self.original_ax.set_title("Nenhum mapa carregado")
            self.original_ax.text(0.5, 0.5, "Clique em 'Carregar Mapa Original'", 
                                 ha='center', va='center', transform=self.original_ax.transAxes)
            self.original_canvas.draw()
        except Exception as e:
            print(f"Erro ao limpar gráfico original: {e}")
    
    def clear_result_plot(self):
        """Limpa completamente o gráfico de resultado."""
        
        try:
            # Limpar axes
            self.result_ax.clear()
            
            # Remover TODAS as colorbars de forma mais agressiva
            for child in list(self.result_fig.get_children()):
                try:
                    if hasattr(child, 'remove'):
                        child_type = str(type(child)).lower()
                        if 'colorbar' in child_type or 'colorbar' in str(child):
                            child.remove()
                except:
                    pass  # Ignorar erros de remoção
            
            # Remover também do axes
            for child in list(self.result_ax.get_children()):
                try:
                    if hasattr(child, 'remove'):
                        child_type = str(type(child)).lower()
                        if 'colorbar' in child_type or 'colorbar' in str(child):
                            child.remove()
                except:
                    pass  # Ignorar erros de remoção
            
            # Resetar limites dos eixos
            self.result_ax.set_xlim(0, 1)
            self.result_ax.set_ylim(0, 1)
            
            # Remover eixos desnecessários
            self.result_ax.set_xticks([])
            self.result_ax.set_yticks([])
            
            self.result_ax.set_title("Nenhum resultado carregado")
            self.result_ax.text(0.5, 0.5, "Clique em 'Gerar Super-Resolução'", 
                               ha='center', va='center', transform=self.result_ax.transAxes)
            self.result_canvas.draw()
        except Exception as e:
            print(f"Erro ao limpar gráfico de resultado: {e}")
    
    def create_shaded_relief(self, elevation_data, azimuth=315, altitude=45, z_factor=1.0):
        """
        Cria sombreamento (shaded relief) a partir dos dados de elevação.
        
        Args:
            elevation_data: Array 2D com dados de elevação
            azimuth: Ângulo de azimute da luz (0-360 graus)
            altitude: Ângulo de altitude da luz (0-90 graus)
            z_factor: Fator de exagero vertical
        """
        
        # Converter para float32
        elevation = elevation_data.astype(np.float32)
        
        # Calcular gradientes (derivadas parciais)
        grad_x, grad_y = np.gradient(elevation * z_factor)
        
        # Converter ângulos para radianos
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        # Calcular componentes da luz
        light_x = np.sin(azimuth_rad) * np.cos(altitude_rad)
        light_y = np.cos(azimuth_rad) * np.cos(altitude_rad)
        light_z = np.sin(altitude_rad)
        
        # Normalizar gradientes
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + 1)
        grad_x_norm = grad_x / grad_magnitude
        grad_y_norm = grad_y / grad_magnitude
        grad_z_norm = 1 / grad_magnitude
        
        # Calcular sombreamento (produto escalar)
        shaded = (light_x * grad_x_norm + 
                 light_y * grad_y_norm + 
                 light_z * grad_z_norm)
        
        # Normalizar para 0-1
        shaded = (shaded - shaded.min()) / (shaded.max() - shaded.min())
        
        # Combinar com dados originais para preservar informações de elevação
        # Usar sombreamento como intensidade e elevação como matiz
        combined = np.stack([elevation, elevation, elevation, shaded], axis=-1)
        
        # Normalizar elevação para 0-1
        elevation_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        
        # Criar imagem RGB com sombreamento
        rgb_image = np.zeros((elevation.shape[0], elevation.shape[1], 3))
        
        # Usar elevação para matiz (H) e sombreamento para valor (V)
        # Converter para HSV e depois para RGB
        h = elevation_norm  # Matiz baseada na elevação
        s = np.ones_like(h)  # Saturação máxima
        v = shaded  # Valor baseado no sombreamento
        
        # Converter HSV para RGB
        rgb_image = self.hsv_to_rgb(h, s, v)
        
        return rgb_image
    
    def hsv_to_rgb(self, h, s, v):
        """Converte HSV para RGB."""
        
        h = h * 6.0  # Escalar para 0-6
        i = np.floor(h).astype(int)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        rgb = np.zeros((h.shape[0], h.shape[1], 3))
        
        # Casos para cada setor do círculo HSV
        mask0 = (i % 6 == 0)
        mask1 = (i % 6 == 1)
        mask2 = (i % 6 == 2)
        mask3 = (i % 6 == 3)
        mask4 = (i % 6 == 4)
        mask5 = (i % 6 == 5)
        
        rgb[mask0] = np.stack([v[mask0], t[mask0], p[mask0]], axis=-1)
        rgb[mask1] = np.stack([q[mask1], v[mask1], p[mask1]], axis=-1)
        rgb[mask2] = np.stack([p[mask2], v[mask2], t[mask2]], axis=-1)
        rgb[mask3] = np.stack([p[mask3], q[mask3], v[mask3]], axis=-1)
        rgb[mask4] = np.stack([t[mask4], p[mask4], v[mask4]], axis=-1)
        rgb[mask5] = np.stack([v[mask5], p[mask5], q[mask5]], axis=-1)
        
        return rgb

def main():
    """Função principal."""
    
    root = tk.Tk()
    app = MapVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()