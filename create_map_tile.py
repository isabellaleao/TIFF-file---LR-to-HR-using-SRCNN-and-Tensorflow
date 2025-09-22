#!/usr/bin/env python3
"""
Script para criar um tile (mosaico) dos mapas usados no projeto de super-resolução.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

def load_geotiff_data(filepath, max_size=1000):
    """
    Carrega dados de um arquivo GeoTIFF e redimensiona se necessário.
    """
    try:
        ds = gdal.Open(filepath)
        if ds is None:
            print(f"❌ Não foi possível abrir: {filepath}")
            return None, None, None
        
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        nodata_value = band.GetNoDataValue()
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        
        # Criar máscara de dados válidos
        valid_mask = ~np.isnan(data) & ~np.isinf(data)
        if nodata_value is not None:
            valid_mask = valid_mask & (data != nodata_value)
        
        # Redimensionar se muito grande
        if data.shape[0] > max_size or data.shape[1] > max_size:
            scale = min(max_size / data.shape[0], max_size / data.shape[1])
            new_height = int(data.shape[0] * scale)
            new_width = int(data.shape[1] * scale)
            
            # Usar interpolação bilinear para redimensionar
            from scipy.ndimage import zoom
            data_resized = zoom(data, (scale, scale), order=1)
            valid_mask_resized = zoom(valid_mask.astype(float), (scale, scale), order=1) > 0.5
        else:
            data_resized = data
            valid_mask_resized = valid_mask
        
        # Aplicar máscara
        data_masked = data_resized.copy()
        data_masked[~valid_mask_resized] = np.nan
        
        ds = None
        return data_masked, geotransform, projection
        
    except Exception as e:
        print(f"❌ Erro ao carregar {filepath}: {e}")
        return None, None, None

def create_map_tile():
    """
    Cria um tile com os mapas principais do projeto.
    """
    print("🗺️ CRIANDO TILE DOS MAPAS DO PROJETO")
    print("=" * 50)
    
    # Definir mapas para incluir no tile
    maps = [
        {
            'file': 'data/images/ANADEM_AricanduvaBufferUTM.tif',
            'title': 'ANADEM (30m/pixel)',
            'description': 'Modelo Digital de Terreno\nResolução: 30m'
        },
        {
            'file': 'data/images/MDTGeosampa_AricanduvaBufferUTM.tif',
            'title': 'GEOSAMPA (0.5m/pixel)',
            'description': 'Modelo Digital de Terreno\nResolução: 0.5m'
        },
        {
            'file': 'geospatial_output/anadem_super_resolution_3x.tif',
            'title': 'Super-Resolução 3x (10m/pixel)',
            'description': 'SRCNN + Bilinear\nResolução: 10m'
        },
        {
            'file': 'geospatial_output/anadem_hybrid_super_resolution_3x.tif',
            'title': 'Híbrido 3x (10m/pixel)',
            'description': 'Pipeline Híbrido\nResolução: 10m'
        }
    ]
    
    # Verificar quais mapas existem
    available_maps = []
    for map_info in maps:
        if os.path.exists(map_info['file']):
            available_maps.append(map_info)
            print(f"✅ {map_info['title']}")
        else:
            print(f"❌ {map_info['title']} - Arquivo não encontrado")
    
    if len(available_maps) == 0:
        print("❌ Nenhum mapa disponível para criar o tile")
        return
    
    # Carregar dados dos mapas
    map_data = []
    for map_info in available_maps:
        print(f"📊 Carregando {map_info['title']}...")
        data, geotransform, projection = load_geotiff_data(map_info['file'])
        if data is not None:
            map_data.append({
                'data': data,
                'title': map_info['title'],
                'description': map_info['description'],
                'geotransform': geotransform,
                'projection': projection
            })
    
    if len(map_data) == 0:
        print("❌ Nenhum mapa foi carregado com sucesso")
        return
    
    # Configurar layout do tile
    n_maps = len(map_data)
    if n_maps == 1:
        rows, cols = 1, 1
    elif n_maps == 2:
        rows, cols = 1, 2
    elif n_maps == 3:
        rows, cols = 1, 3
    elif n_maps == 4:
        rows, cols = 2, 2
    else:
        rows = int(np.ceil(n_maps / 3))
        cols = 3
    
    # Criar figura
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_maps == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Configurar colormap
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    cmap = LinearSegmentedColormap.from_list('terrain', colors, N=256)
    
    # Plotar cada mapa
    for i, map_info in enumerate(map_data):
        ax = axes[i]
        data = map_info['data']
        
        # Calcular estatísticas
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 0, 1
        
        # Plotar mapa
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        
        # Configurar título e labels
        ax.set_title(f"{map_info['title']}\n{map_info['description']}", 
                    fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('Pixels (X)', fontsize=8)
        ax.set_ylabel('Pixels (Y)', fontsize=8)
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Elevação (m)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Adicionar informações de resolução
        if map_info['geotransform'] is not None:
            pixel_size = abs(map_info['geotransform'][1])
            ax.text(0.02, 0.98, f'Resolução: {pixel_size:.1f}m/pixel', 
                   transform=ax.transAxes, fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   verticalalignment='top')
        
        # Adicionar informações de dimensões
        ax.text(0.98, 0.02, f'{data.shape[1]}×{data.shape[0]}', 
               transform=ax.transAxes, fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
               verticalalignment='bottom', horizontalalignment='right')
        
        # Configurar ticks
        ax.tick_params(labelsize=7)
        
        # Adicionar grid sutil
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Ocultar eixos vazios
    for i in range(n_maps, len(axes)):
        axes[i].set_visible(False)
    
    # Configurar layout geral
    plt.suptitle('🗺️ MAPAS DO PROJETO DE SUPER-RESOLUÇÃO GEOGRÁFICA', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Salvar tile
    output_path = 'geospatial_output/map_tile_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Tile salvo: {output_path}")
    
    # Mostrar estatísticas
    print("\n📊 ESTATÍSTICAS DOS MAPAS:")
    print("-" * 40)
    for map_info in map_data:
        data = map_info['data']
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(f"{map_info['title']}:")
            print(f"  Dimensões: {data.shape[1]}×{data.shape[0]} pixels")
            print(f"  Elevação: {np.min(valid_data):.1f}m - {np.max(valid_data):.1f}m")
            print(f"  Média: {np.mean(valid_data):.1f}m")
            print(f"  Std: {np.std(valid_data):.1f}m")
            print()
    
    plt.show()

def create_detailed_comparison():
    """
    Cria uma comparação detalhada entre os mapas.
    """
    print("\n🔍 CRIANDO COMPARAÇÃO DETALHADA")
    print("=" * 40)
    
    # Carregar ANADEM original
    anadem_data, _, _ = load_geotiff_data('data/images/ANADEM_AricanduvaBufferUTM.tif', max_size=500)
    if anadem_data is None:
        print("❌ Não foi possível carregar ANADEM")
        return
    
    # Carregar resultado de super-resolução
    sr_data, _, _ = load_geotiff_data('geospatial_output/anadem_super_resolution_3x.tif', max_size=500)
    if sr_data is None:
        print("❌ Não foi possível carregar resultado de super-resolução")
        return
    
    # Criar figura de comparação
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ANADEM original
    ax1 = axes[0, 0]
    valid_anadem = anadem_data[~np.isnan(anadem_data)]
    vmin, vmax = np.percentile(valid_anadem, [2, 98])
    im1 = ax1.imshow(anadem_data, cmap='terrain', vmin=vmin, vmax=vmax)
    ax1.set_title('ANADEM Original (30m/pixel)', fontweight='bold')
    ax1.set_xlabel('Pixels (X)')
    ax1.set_ylabel('Pixels (Y)')
    plt.colorbar(im1, ax=ax1, label='Elevação (m)')
    
    # Super-resolução
    ax2 = axes[0, 1]
    valid_sr = sr_data[~np.isnan(sr_data)]
    vmin, vmax = np.percentile(valid_sr, [2, 98])
    im2 = ax2.imshow(sr_data, cmap='terrain', vmin=vmin, vmax=vmax)
    ax2.set_title('Super-Resolução 3x (10m/pixel)', fontweight='bold')
    ax2.set_xlabel('Pixels (X)')
    ax2.set_ylabel('Pixels (Y)')
    plt.colorbar(im2, ax=ax2, label='Elevação (m)')
    
    # Histograma comparativo
    ax3 = axes[1, 0]
    ax3.hist(valid_anadem, bins=50, alpha=0.7, label='ANADEM Original', color='blue', density=True)
    ax3.hist(valid_sr, bins=50, alpha=0.7, label='Super-Resolução 3x', color='red', density=True)
    ax3.set_xlabel('Elevação (m)')
    ax3.set_ylabel('Densidade')
    ax3.set_title('Distribuição de Elevações')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Diferença
    ax4 = axes[1, 1]
    # Redimensionar ANADEM para mesma resolução da super-resolução
    from scipy.ndimage import zoom
    scale_factor = sr_data.shape[0] / anadem_data.shape[0]
    anadem_resized = zoom(anadem_data, scale_factor, order=1)
    
    # Calcular diferença
    diff = sr_data - anadem_resized
    valid_diff = diff[~np.isnan(diff)]
    
    im4 = ax4.imshow(diff, cmap='RdBu_r', vmin=np.percentile(valid_diff, 2), 
                     vmax=np.percentile(valid_diff, 98))
    ax4.set_title('Diferença (SR - ANADEM)', fontweight='bold')
    ax4.set_xlabel('Pixels (X)')
    ax4.set_ylabel('Pixels (Y)')
    plt.colorbar(im4, ax=ax4, label='Diferença (m)')
    
    plt.suptitle('🔍 COMPARAÇÃO DETALHADA - ANADEM vs SUPER-RESOLUÇÃO', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Salvar comparação
    output_path = 'geospatial_output/detailed_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparação detalhada salva: {output_path}")
    
    plt.show()

def main():
    """Função principal."""
    print("🗺️ GERADOR DE TILE DE MAPAS")
    print("=" * 50)
    
    # Criar tile principal
    create_map_tile()
    
    # Criar comparação detalhada
    create_detailed_comparison()
    
    print("\n🎉 Processo concluído!")

if __name__ == "__main__":
    main()
