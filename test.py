#!/usr/bin/env python3
import numpy as np
from osgeo import gdal
import cv2
import os


def _load_model(model_path, scale_factor=3):
    import tensorflow as tf
    # Carrega o modelo salvo apenas para obter os pesos
    fixed = tf.keras.models.load_model(model_path)
    # Reconstrói a arquitetura com dimensões espaciais dinâmicas
    inputs = tf.keras.layers.Input(shape=(None, None, 1))
    x1 = tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    x2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    x3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([x1, x2, x3])
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D(size=(scale_factor, scale_factor), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # Copia os pesos do modelo fixo
    model.set_weights(fixed.get_weights())
    print(f"✅ Pesos carregados de: {model_path}")
    print(f"   Input dinâmico: {model.input_shape} → Output: {model.output_shape}")
    return model


def _load_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    prj = ds.GetProjection()
    ds = None
    if nodata is not None:
        try:
            arr[arr == np.float32(nodata)] = np.nan
        except Exception:
            arr[arr == nodata] = np.nan
    return arr, nodata, gt, prj


def _save_raster(path, arr, gt, prj, nodata=None):
    driver = gdal.GetDriverByName('GTiff')
    h, w = arr.shape
    ds = driver.Create(path, w, h, 1, gdal.GDT_Float32, options=['COMPRESS=LZW','TILED=YES'])
    ds.SetGeoTransform(gt)
    ds.SetProjection(prj)
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(arr.astype(np.float32))
    band.FlushCache()
    ds = None


def main():
    anadem_path = 'data/images/ANADEM_AricanduvaBufferUTM.tif'
    model_path = 'geospatial_output/advanced_model_best.keras'
    norm_params_path = 'geospatial_output/norm_params.npy'
    out_path = 'geospatial_output/ANADEM_model_only_3x.tif'
    scale_factor = 3

    # Carregar modelo e normalização
    model = _load_model(model_path, scale_factor=scale_factor)
    norm = np.load(norm_params_path, allow_pickle=True).item()
    data_min = float(norm['hr_min'])
    data_max = float(norm['hr_max'])

    # Carregar ANADEM inteiro
    anadem, nodata, gt, prj = _load_raster(anadem_path)
    h, w = anadem.shape
    print(f"📊 ANADEM: {w}x{h}")

    # Tratar inválidos globalmente (sem recorte em patches)
    valid_mask = ~np.isnan(anadem) & ~np.isinf(anadem)
    if nodata is not None:
        valid_mask = valid_mask & (anadem != nodata)
    vals = anadem[valid_mask]
    fill_value = float(np.mean(vals)) if vals.size > 0 else 0.0
    anadem_filled = anadem.copy()
    anadem_filled[~valid_mask] = fill_value

    # Normalizar e inferir a imagem inteira (fully-convolutional)
    rng = max(1e-12, (data_max - data_min))
    img_norm = (anadem_filled - data_min) / rng
    img_norm = np.clip(img_norm, 0.0, 1.0)
    inp = img_norm[None, ..., None]
    pred = model.predict(inp, verbose=1)[0, :, :, 0]

    # Desnormalizar
    out_hr = pred * (data_max - data_min) + data_min

    # Propagar máscara de inválidos para a saída em 10 m
    if np.any(~valid_mask):
        mask_hr = np.repeat(np.repeat(valid_mask.astype(np.uint8), scale_factor, axis=0), scale_factor, axis=1) > 0
        out_hr[~mask_hr] = np.nan

    # Geotransform para 10 m
    gt_hr = list(gt)
    gt_hr[1] = gt[1] / scale_factor
    gt_hr[5] = gt[5] / scale_factor

    # Salvar TIFF final (área inteira)
    _save_raster(out_path, out_hr, tuple(gt_hr), prj, nodata=nodata)
    print(f"✅ Modelo-only salvo: {out_path}")


if __name__ == '__main__':
    main()

