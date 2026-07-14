"""
================================================================================
  MONITOREO DE RUIDO AMBIENTAL URBANO
  Análisis Espectral, Estadístico y Detección de Anomalías

  ⚠️  AVISO: Proyecto de portafolio personal con datos 100% SINTÉTICOS.
  No guarda relación con proyectos reales, clientes, metodologías propietarias
  ni datos confidenciales de empleadores actuales o anteriores.
  Las normativas mencionadas (ISO 1996, DS 38/11 Chile, Ley 1540 CABA) son
  de conocimiento público y su aplicación aquí es meramente ilustrativa.

  Autor  : Alberto Lopardo
  LinkedIn: linkedin.com/in/alberto-lopardo-acustica
  Ref.   : ISO 1996-1:2016 / ISO 1996-2:2017
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibilidad ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── Paleta de colores ─────────────────────────────────────────────────────────
C_NAVY   = '#1B3A5C'
C_BLUE   = '#2E7DBF'
C_CYAN   = '#5DB8E0'
C_GREEN  = '#2E8B57'
C_ORANGE = '#E07B39'
C_RED    = '#C0392B'
C_LIGHT  = '#F4F7FA'
C_GRID   = '#DDE3EA'
C_MID    = '#4A4A4A'

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        C_GRID,
    'grid.linewidth':    0.6,
    'axes.labelcolor':   C_NAVY,
    'xtick.color':       C_MID,
    'ytick.color':       C_MID,
    'figure.facecolor':  'white',
})

FONT_NOTE = dict(fontsize=7.5, color='gray', style='italic')
NOTE_TEXT = ('⚠  Datos sintéticos generados para demostración técnica — '
             'no representan mediciones reales ni datos de ningún cliente o empleador.')

# ==============================================================================
# 1. PARÁMETROS DE SIMULACIÓN
# ==============================================================================

# Período de monitoreo: 7 días continuos
DIAS              = 7
HORAS_TOTALES     = DIAS * 24
FECHA_INICIO      = datetime(2024, 10, 1, 0, 0, 0)

# Límites normativos — zona residencial/mixta
# Ref: ISO 1996 / DS 38/11 Chile / Res. SRT 85/2012 Argentina
LIMITE_LEQ_DIA   = 55.0   # dBA  07:00–22:00
LIMITE_LEQ_NOCH  = 50.0   # dBA  22:00–07:00
LIMITE_LMAX_NOCH = 65.0   # dBA  (eventos impulsivos nocturnos)

# Bandas de tercio de octava (Hz) — rango ISO 1996
TOB_FREQS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000
])

# ==============================================================================
# 2. GENERACIÓN DE DATOS SINTÉTICOS
# ==============================================================================

fechas = [FECHA_INICIO + timedelta(hours=i) for i in range(HORAS_TOTALES)]
df = pd.DataFrame({'FechaHora': fechas})
df['Hora']     = df['FechaHora'].dt.hour
df['Dia']      = df['FechaHora'].dt.day
df['Nocturno'] = ((df['Hora'] >= 22) | (df['Hora'] < 7)).astype(int)

# Perfil diurno realista: dos picos de tráfico (mañana/tarde) + valle nocturno
def perfil_trafico(hora):
    base  = 50.0
    p_man = 5.0  * np.exp(-((hora - 8.0)  / 1.5) ** 2)
    p_tar = 4.5  * np.exp(-((hora - 18.0) / 1.5) ** 2)
    noche = -8.0 * (1/(1+np.exp(-(hora-7))) - 1/(1+np.exp(-(hora-22))))
    return base + p_man + p_tar + noche

nivel_base = df['Hora'].apply(perfil_trafico).values

# L90 — ruido de fondo (diurno vs nocturno)
df['L90'] = np.where(
    df['Nocturno'] == 1,
    rng.normal(38, 1.8, HORAS_TOTALES),
    rng.normal(46, 2.5, HORAS_TOTALES)
)

# Leq, L10, Lmax coherentes con el perfil base
ruido_leq  = rng.normal(0, 1.5, HORAS_TOTALES)
ruido_leq  = np.convolve(ruido_leq, np.ones(3)/3, mode='same')
df['Leq']  = nivel_base + ruido_leq
df['L10']  = df['Leq'] + rng.normal(3.5, 0.8, HORAS_TOTALES)
lmax_delta = np.where(df['Nocturno']==1, rng.normal(8.5,1.2,HORAS_TOTALES), rng.normal(13.0,2.0,HORAS_TOTALES))
df['Lmax'] = np.maximum(df['L90'] + lmax_delta, df['Leq'] + rng.normal(2.5, 0.5, HORAS_TOTALES))

# ── Inyección de eventos anómalos ─────────────────────────────────────────────
eventos = [
    # (dia, hora, delta_lmax, delta_leq, descripción)
    (3, 3,  +18, +5, 'Recolección nocturna de residuos'),
    (5, 2,  +22, +7, 'Evento social / música amplificada'),
    (6, 6,  +15, +3, 'Inicio anticipado de obra'),
]
for dia, hora, dlmax, dleq, _ in eventos:
    idx = df[(df['Hora'] == hora) & (df['Dia'] == dia)].index
    if len(idx):
        df.loc[idx[0], 'Lmax'] += dlmax
        df.loc[idx[0], 'Leq']  += dleq

# ==============================================================================
# 3. DETECCIÓN DE ANOMALÍAS (criterios ISO 1996-2)
# ==============================================================================

# A — Superación del límite normativo de Leq
df['Supera_Leq'] = (
    ((df['Nocturno'] == 1) & (df['Leq']  > LIMITE_LEQ_NOCH)) |
    ((df['Nocturno'] == 0) & (df['Leq']  > LIMITE_LEQ_DIA))
)

# B — Evento impulsivo nocturno: Lmax - L90 > 15 dB (criterio ISO 1996-2)
df['Impulsivo'] = (df['Nocturno'] == 1) & ((df['Lmax'] - df['L90']) > 20)

# C — Superación de Lmax absoluto nocturno
df['Supera_Lmax'] = (df['Nocturno'] == 1) & (df['Lmax'] > LIMITE_LMAX_NOCH)

df['Es_Anomalia'] = df['Supera_Leq'] | df['Impulsivo'] | df['Supera_Lmax']
anomalias = df[df['Es_Anomalia']]

# ==============================================================================
# 4. DATOS ESPECTRALES — TERCIO DE OCTAVA
# ==============================================================================

def espectro_tob(perfil='dia'):
    shape = 72 - 10 * np.log10(TOB_FREQS / 63)
    offset = {'dia': 0, 'noche': -9, '24h': -4}[perfil]
    noise  = rng.normal(0, 0.9, len(TOB_FREQS))
    return np.clip(shape + offset + noise, 28, 80)

tob_dia   = espectro_tob('dia')
tob_noche = espectro_tob('noche')
tob_24h   = espectro_tob('24h')

# Mapa de calor hora × frecuencia (promedio por hora sobre 7 días)
horas_unicas = np.arange(24)
heatmap = np.zeros((24, len(TOB_FREQS)))
for h in horas_unicas:
    offset = perfil_trafico(h) - 50
    shape  = 72 - 10 * np.log10(TOB_FREQS / 63) + offset
    heatmap[h] = np.clip(shape + rng.normal(0, 1.2, len(TOB_FREQS)), 28, 82)

# ==============================================================================
# 5. FIGURA 1 — SERIE TEMPORAL 7 DÍAS (Leq, Lmax, anomalías)
# ==============================================================================

fig1, ax = plt.subplots(figsize=(15, 5.5))

ax.plot(df['FechaHora'], df['Leq'],  color=C_NAVY,   lw=1.8,         label='Leq (dBA)',  zorder=3)
ax.plot(df['FechaHora'], df['Lmax'], color=C_ORANGE,  lw=1.2, alpha=.8, label='Lmax (dBA)', zorder=3)
ax.plot(df['FechaHora'], df['L90'],  color=C_GREEN,   lw=1.0, ls=':',  label='L90 – ruido de fondo', zorder=3)

# Anomalías
ax.scatter(anomalias['FechaHora'], anomalias['Lmax'],
           color=C_RED, zorder=6, s=55, label='Anomalía detectada', marker='D')

# Límites normativos
ax.axhline(LIMITE_LEQ_DIA,  color=C_RED, lw=1.5, ls='--', alpha=.75,
           label=f'Límite Leq diurno ({LIMITE_LEQ_DIA} dBA)')
ax.axhline(LIMITE_LEQ_NOCH, color=C_RED, lw=1.5, ls=':',  alpha=.75,
           label=f'Límite Leq nocturno ({LIMITE_LEQ_NOCH} dBA)')

# Etiquetas de eventos
for dia, hora, _, _, desc in eventos:
    t = FECHA_INICIO + timedelta(days=dia-1, hours=hora)
    fila = df[(df['Dia'] == dia) & (df['Hora'] == hora)]
    if len(fila):
        y = fila['Lmax'].values[0]
        ax.annotate(desc, xy=(t, y), xytext=(t + timedelta(hours=6), y + 4),
                    arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.1),
                    fontsize=8, color=C_RED)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m\n%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
ax.set_xlabel('Fecha y Hora', fontsize=11, fontweight='bold')
ax.set_ylabel('Nivel Sonoro (dBA)', fontsize=11, fontweight='bold')
ax.set_title('Monitoreo Continuo de Ruido Ambiental — Serie Temporal 7 Días\n'
             'Zona residencial/mixta · Detección de eventos atípicos · ISO 1996-2:2017',
             fontsize=12, fontweight='bold', color=C_NAVY, pad=12)
ax.legend(loc='upper right', fontsize=8.5, framealpha=.92, ncol=3)
ax.set_ylim(28, 88)
fig1.text(0.01, 0.01, NOTE_TEXT, **FONT_NOTE)
plt.tight_layout()
fig1.savefig('/home/claude/fig1_serie_7dias.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig 1 — Serie temporal 7 días: OK")

# ==============================================================================
# 6. FIGURA 2 — PERFIL HORARIO PROMEDIO (24 h)
# ==============================================================================

fig2, ax = plt.subplots(figsize=(13, 5))

horas_plot = np.arange(24)
leq_hora  = df.groupby('Hora')['Leq'].mean().values
l10_hora  = df.groupby('Hora')['L10'].mean().values
l90_hora  = df.groupby('Hora')['L90'].mean().values
lmax_hora = df.groupby('Hora')['Lmax'].mean().values
lmin_hora = df.groupby('Hora')['Leq'].min().values

ax.fill_between(horas_plot, lmin_hora, lmax_hora, alpha=.10, color=C_BLUE, label='Rango Lmin–Lmax')
ax.fill_between(horas_plot, l90_hora,  l10_hora,  alpha=.20, color=C_BLUE, label='Banda L10–L90')
ax.plot(horas_plot, l10_hora,  color=C_BLUE,   lw=1.4, ls='--', label='L10')
ax.plot(horas_plot, l90_hora,  color=C_GREEN,  lw=1.4, ls=':',  label='L90')
ax.plot(horas_plot, leq_hora,  color=C_NAVY,   lw=2.3,          label='Leq promedio (7 días)')
ax.plot(horas_plot, lmax_hora, color=C_ORANGE, lw=1.2, ls='-.', label='Lmax')

ax.axhline(LIMITE_LEQ_DIA,  color=C_RED, lw=1.5, ls='--', alpha=.8)
ax.axhline(LIMITE_LEQ_NOCH, color=C_RED, lw=1.5, ls=':',  alpha=.8)
ax.text(23.1, LIMITE_LEQ_DIA  + .5, f'Límite diurno {LIMITE_LEQ_DIA} dBA',   color=C_RED, fontsize=7.5)
ax.text(23.1, LIMITE_LEQ_NOCH + .5, f'Límite nocturno {LIMITE_LEQ_NOCH} dBA', color=C_RED, fontsize=7.5)

ax.axvspan(0,  7, alpha=.06, color='navy')
ax.axvspan(22, 24, alpha=.06, color='navy')
ax.text(3.5, 30, 'Nocturno', ha='center', color='navy', fontsize=8.5, alpha=.65)
ax.text(23,  30, 'Noc.',     ha='center', color='navy', fontsize=8,   alpha=.65)

ax.set_xlim(0, 24)
ax.set_ylim(27, 82)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])
ax.set_xlabel('Hora del día', fontsize=11, fontweight='bold')
ax.set_ylabel('Nivel de presión sonora (dBA)', fontsize=11, fontweight='bold')
ax.set_title('Perfil Horario Promedio — Niveles Estadísticos (Leq, L10, L90, Lmax)\n'
             'Promedio sobre 7 días de monitoreo · Zona residencial/mixta · ISO 1996-2:2017',
             fontsize=12, fontweight='bold', color=C_NAVY, pad=12)
ax.legend(loc='upper right', fontsize=8.5, framealpha=.92, ncol=2)
fig2.text(0.01, 0.01, NOTE_TEXT, **FONT_NOTE)
plt.tight_layout()
fig2.savefig('/home/claude/fig2_perfil_horario.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig 2 — Perfil horario 24 h: OK")

# ==============================================================================
# 7. FIGURA 3 — ESPECTRO DE TERCIO DE OCTAVA
# ==============================================================================

fig3, ax = plt.subplots(figsize=(13, 5.5))

x = np.arange(len(TOB_FREQS))
w = 0.27
ax.bar(x - w,  tob_dia,   width=w, color=C_NAVY,   alpha=.88, label='Período diurno (07–22 h)',   zorder=3)
ax.bar(x,      tob_noche, width=w, color=C_BLUE,   alpha=.75, label='Período nocturno (22–07 h)', zorder=3)
ax.bar(x + w,  tob_24h,   width=w, color=C_GREEN,  alpha=.70, label='Promedio 24 h',              zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([str(f) for f in TOB_FREQS], rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Frecuencia central de banda (Hz)', fontsize=11, fontweight='bold')
ax.set_ylabel('Nivel de presión sonora (dBA)', fontsize=11, fontweight='bold')
ax.set_title('Espectro de Ruido — Bandas de Tercio de Octava\n'
             'Zona residencial/mixta · Datos sintéticos · ISO 1996-1:2016',
             fontsize=12, fontweight='bold', color=C_NAVY, pad=12)
ax.set_ylim(28, 86)

idx_1k = np.where(TOB_FREQS == 1000)[0][0]
ax.axvline(x=idx_1k, color='gray', lw=.8, ls=':', alpha=.6)
ax.text(idx_1k + .2, 82, '1 kHz', color='gray', fontsize=8)

ax.annotate('Dominancia graves\n(tráfico rodado)',
            xy=(3, tob_dia[3]), xytext=(6, 81),
            arrowprops=dict(arrowstyle='->', color=C_NAVY, lw=1.2),
            fontsize=8.5, color=C_NAVY)
ax.legend(fontsize=9.5, framealpha=.92)
fig3.text(0.01, 0.01, NOTE_TEXT, **FONT_NOTE)
plt.tight_layout()
fig3.savefig('/home/claude/fig3_espectro_tob.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig 3 — Espectro TOB: OK")

# ==============================================================================
# 8. FIGURA 4 — MAPA DE CALOR hora × frecuencia
# ==============================================================================

fig4, ax = plt.subplots(figsize=(13, 5.5))

cmap = LinearSegmentedColormap.from_list('acoustic',
    ['#0D2B45', '#1B3A5C', '#2E7DBF', '#5DB8E0', '#A8D8EA',
     '#F9E784', '#F4A261', '#E07B39', '#C0392B'], N=256)

im = ax.imshow(heatmap.T, aspect='auto', origin='lower', cmap=cmap,
               vmin=36, vmax=80, extent=[0, 24, 0, len(TOB_FREQS)])

cbar = fig4.colorbar(im, ax=ax, pad=.02)
cbar.set_label('Nivel (dBA)', fontsize=10, color=C_NAVY)

ax.axvline(x=7,  color='white', lw=1.8, ls='--', alpha=.75)
ax.axvline(x=22, color='white', lw=1.8, ls='--', alpha=.75)
ax.text(3.5, len(TOB_FREQS)-.8, 'Nocturno', color='white', fontsize=9, fontweight='bold', ha='center')
ax.text(14.5, len(TOB_FREQS)-.8, 'Diurno',   color='white', fontsize=9, fontweight='bold', ha='center')
ax.text(23.1, len(TOB_FREQS)-.8, 'Noc.',     color='white', fontsize=8, fontweight='bold', ha='center')

ax.set_xticks(np.arange(0, 25, 2))
ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 2)])
ax.set_yticks(np.arange(len(TOB_FREQS)))
ax.set_yticklabels([str(f) for f in TOB_FREQS], fontsize=7.5)
ax.set_xlabel('Hora del día (promedio 7 días)', fontsize=11, fontweight='bold')
ax.set_ylabel('Banda de tercio de octava (Hz)', fontsize=11, fontweight='bold')
ax.set_title('Mapa de Calor — Distribución Espectro-Temporal del Ruido Ambiental\n'
             'Zona residencial/mixta · Datos sintéticos · ISO 1996-2:2017',
             fontsize=12, fontweight='bold', color=C_NAVY, pad=12)
fig4.text(0.01, 0.01, NOTE_TEXT, **FONT_NOTE)
plt.tight_layout()
fig4.savefig('/home/claude/fig4_heatmap.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig 4 — Mapa de calor: OK")

# ==============================================================================
# 9. FIGURA 5 — DASHBOARD RESUMEN (4 paneles)
# ==============================================================================

fig5 = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, figure=fig5, hspace=.42, wspace=.35)

is_day   = (df['Nocturno'] == 0)
is_night = (df['Nocturno'] == 1)

# ── Panel A: Excedencia horaria ───────────────────────────────────────────────
ax_a = fig5.add_subplot(gs[0, 0])
limite_hora = np.where(df.groupby('Hora')['Nocturno'].mean() > .5,
                       LIMITE_LEQ_NOCH, LIMITE_LEQ_DIA)
leq_hora_plot = df.groupby('Hora')['Leq'].mean().values
excedencia = leq_hora_plot - limite_hora
colores_exc = [C_RED if e > 0 else C_GREEN for e in excedencia]
ax_a.bar(horas_unicas, excedencia, color=colores_exc, alpha=.85, zorder=3)
ax_a.axhline(0, color='black', lw=1.0)
ax_a.set_xlabel('Hora del día', fontsize=10, fontweight='bold')
ax_a.set_ylabel('Excedencia (dB)', fontsize=10, fontweight='bold')
ax_a.set_title('A — Excedencia del Límite Normativo', fontsize=10, fontweight='bold', color=C_NAVY)
ax_a.set_xlim(-.5, 23.5)
ax_a.xaxis.set_major_locator(ticker.MultipleLocator(4))
n_exc = sum(1 for e in excedencia if e > 0)
ax_a.text(.98, .97, f'{n_exc}/24 horas en excedencia',
          transform=ax_a.transAxes, ha='right', va='top',
          fontsize=9, color=C_RED, fontweight='bold')

# ── Panel B: Estadísticos diurno vs nocturno ─────────────────────────────────
ax_b = fig5.add_subplot(gs[0, 1])
etiquetas = ['Leq', 'L10', 'L90', 'Lmax']
cols_map  = {'Leq': 'Leq', 'L10': 'L10', 'L90': 'L90', 'Lmax': 'Lmax'}
dia_vals   = [df.loc[is_day,   cols_map[e]].mean() for e in etiquetas]
noche_vals = [df.loc[is_night, cols_map[e]].mean() for e in etiquetas]
x_s = np.arange(len(etiquetas))
ax_b.bar(x_s - .2, dia_vals,   width=.38, color=C_NAVY,  alpha=.88, label='Diurno  (07–22 h)')
ax_b.bar(x_s + .2, noche_vals, width=.38, color=C_BLUE,  alpha=.75, label='Nocturno (22–07 h)')
ax_b.axhline(LIMITE_LEQ_DIA,  color=C_RED, ls='--', lw=1.4, alpha=.8,
             label=f'Límite diurno {LIMITE_LEQ_DIA} dBA')
ax_b.axhline(LIMITE_LEQ_NOCH, color=C_RED, ls=':',  lw=1.4, alpha=.8,
             label=f'Límite nocturno {LIMITE_LEQ_NOCH} dBA')
ax_b.set_xticks(x_s)
ax_b.set_xticklabels(etiquetas, fontsize=10)
ax_b.set_ylabel('dBA', fontsize=10, fontweight='bold')
ax_b.set_title('B — Estadísticos Diurno vs. Nocturno', fontsize=10, fontweight='bold', color=C_NAVY)
ax_b.legend(fontsize=8, framealpha=.92)
ax_b.set_ylim(28, 76)

# ── Panel C: Curva Ln (percentiles acumulados) ────────────────────────────────
ax_c = fig5.add_subplot(gs[1, 0])
pct = np.arange(1, 100)
ax_c.plot(pct, np.percentile(df.loc[is_day,   'Leq'], pct), color=C_NAVY,  lw=2., label='Diurno')
ax_c.plot(pct, np.percentile(df.loc[is_night, 'Leq'], pct), color=C_BLUE,  lw=2., label='Nocturno')
ax_c.axhline(LIMITE_LEQ_DIA,  color=C_RED, ls='--', lw=1.3, alpha=.8)
ax_c.axhline(LIMITE_LEQ_NOCH, color=C_RED, ls=':',  lw=1.3, alpha=.8)
for p, label in [(10, 'L10'), (50, 'L50'), (90, 'L90')]:
    ax_c.axvline(p, color='gray', lw=.7, ls=':', alpha=.6)
    ax_c.text(p+.5, 29, label, color='gray', fontsize=8)
ax_c.set_xlabel('Percentil (%)', fontsize=10, fontweight='bold')
ax_c.set_ylabel('Leq (dBA)', fontsize=10, fontweight='bold')
ax_c.set_title('C — Curva de Percentiles Acumulados (Ln)', fontsize=10, fontweight='bold', color=C_NAVY)
ax_c.legend(fontsize=9)
ax_c.set_xlim(0, 100)
ax_c.set_ylim(27, 75)

# ── Panel D: Tabla resumen de anomalías ───────────────────────────────────────
ax_d = fig5.add_subplot(gs[1, 1])
ax_d.axis('off')

CAUSA_MAP = {'Impulsivo': 'Impulsivo', 'Supera_Leq': 'Leq > límite', 'Supera_Lmax': 'Lmax > 65 dBA'}
tabla = [['Fecha / Hora', 'Leq', 'Lmax', 'Causa detectada']]
for _, row in anomalias.iterrows():
    clist = [v for k, v in CAUSA_MAP.items() if row[k]]
    tabla.append([row['FechaHora'].strftime('%d/%m %H:%M'), f"{row['Leq']:.1f}", f"{row['Lmax']:.1f}", ' + '.join(clist)])

col_widths = [0.26, 0.12, 0.12, 0.50]
tbl = ax_d.table(cellText=tabla[1:], colLabels=tabla[0],
                 cellLoc='center', loc='center', bbox=[0, 0, 1, 1], colWidths=col_widths)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#BCC8D4')
    if r == 0:
        cell.set_facecolor(C_NAVY)
        cell.set_text_props(color='white', fontweight='bold', fontsize=8)
    elif r % 2 == 0:
        cell.set_facecolor('#EEF3F8')
    else:
        cell.set_facecolor('white')
    if c == 3 and r > 0:
        cell.set_text_props(color=C_RED, fontweight='bold')

ax_d.set_title('D — Registro de Anomalías Detectadas (7 días)',
               fontsize=10, fontweight='bold', color=C_NAVY, pad=8)

fig5.suptitle('Dashboard — Análisis de Ruido Ambiental Urbano (7 días)\n'
              'Zona residencial/mixta · Datos sintéticos · ISO 1996-1:2016 / ISO 1996-2:2017',
              fontsize=13, fontweight='bold', color=C_NAVY, y=1.01)
fig5.text(0.01, -0.01, NOTE_TEXT, **FONT_NOTE)
plt.savefig('/home/claude/fig5_dashboard.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig 5 — Dashboard: OK")

# ==============================================================================
# 10. REPORTE DE CONSOLA
# ==============================================================================

print()
print("=" * 65)
print("  REPORTE AUTOMÁTICO — DETECCIÓN DE EVENTOS (DATOS SINTÉTICOS)")
print("=" * 65)
print(f"  Período monitoreo : {FECHA_INICIO.strftime('%d/%m/%Y')} — "
      f"{(FECHA_INICIO + timedelta(days=DIAS-1)).strftime('%d/%m/%Y')}  ({DIAS} días)")
print(f"  Horas totales     : {HORAS_TOTALES}")
print(f"  Leq diurno promedio  : {df.loc[is_day,   'Leq'].mean():.1f} dBA  "
      f"(límite {LIMITE_LEQ_DIA} dBA)")
print(f"  Leq nocturno promedio: {df.loc[is_night, 'Leq'].mean():.1f} dBA  "
      f"(límite {LIMITE_LEQ_NOCH} dBA)")
print(f"  Anomalías detectadas : {len(anomalias)}")
print("-" * 65)
for _, row in anomalias.iterrows():
    causas = []
    if row['Impulsivo']:   causas.append('Impulsivo')
    if row['Supera_Leq']:  causas.append('Leq > límite')
    if row['Supera_Lmax']: causas.append('Lmax nocturno')
    print(f"  📅 {row['FechaHora'].strftime('%d/%m/%Y %H:%M')}  |  "
          f"Leq {row['Leq']:.1f} dBA  Lmax {row['Lmax']:.1f} dBA  |  "
          f"{', '.join(causas)}")
print("=" * 65)

# Opcional: exportar dataset sintético
# df.to_csv('monitoreo_acustico_sintetico_7dias.csv', index=False)
# print("\n✅ Dataset exportado a CSV.")
