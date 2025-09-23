import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import calendar
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------
# Estatísticas suficientes para percentual de atrasos
# -----------------------------------------

# Total de voos
# Total de voos atrasados (ARRIVAL_DELAY > 10)
# Percentual = atrasados / total

# -----------------------------------------
# 1. Função getStats
# -----------------------------------------

def getStats(chunk):
    df = chunk[
        (chunk["AIRLINE"].isin(["AA"])) &
        (~chunk["ARRIVAL_DELAY"].isna()) &
        (~chunk["YEAR"].isna()) &
        (~chunk["MONTH"].isna()) &
        (~chunk["DAY"].isna())
    ]

    grouped = df.groupby(["YEAR", "MONTH", "DAY", "AIRLINE"], as_index=False).agg(
        n_total=("ARRIVAL_DELAY", "count"),
        n_atrasados=("ARRIVAL_DELAY", lambda x: (x > 10).sum())
    )
    return grouped

# -----------------------------------------
# 2. Leitura em chunks
# -----------------------------------------
all_chunks = []
arquivo = "//smb/ra277200/Downloads/flights.csv.zip"
chunksize = 100_000

with zipfile.ZipFile(arquivo) as z:
    nome_csv = z.namelist()[0]
    for chunk in pd.read_csv(
        z.open(nome_csv),
        usecols=["AIRLINE", "YEAR", "MONTH", "DAY", "ARRIVAL_DELAY"],
        chunksize=chunksize
    ):
        all_chunks.append(getStats(chunk))

df_raw = pd.concat(all_chunks, ignore_index=True)

# -----------------------------------------
# 3. Função computeStats
# -----------------------------------------
def computeStats(input_df):
    df = input_df.groupby(["AIRLINE", "DAY", "MONTH", "YEAR"], as_index=False).agg({
        "n_total": "sum",
        "n_atrasados": "sum"
    })
    df["Perc"] = df["n_atrasados"] / df["n_total"]
    df["Data"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]), errors="coerce")
    return df[["AIRLINE", "Data", "Perc"]]

df_stats = computeStats(df_raw)


# -----------------------------------------
# 4. Função estilo ggcal: um calendário com os 12 meses juntos
# -----------------------------------------

def baseCalendario_mes_completo(stats, cia, year=2015):
    df = stats[stats["AIRLINE"] == cia].copy()
    df = df.set_index("Data")

    meses_pt = [
        "janeiro", "fevereiro", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
    ]
    dias_semana = ["S", "M", "T", "W", "T", "F", "S"]

    fig, axes = plt.subplots(4, 3, figsize=(14, 10))
    fig.suptitle(f"{cia} - Percentual de Atrasos por Dia - 2015", fontsize=16)

    # Paleta de cores customizada
    pal = LinearSegmentedColormap.from_list("custom", ["#4575b4", "#d73027"])

    for month in range(1, 13):
        row = (month - 1) // 3
        col = (month - 1) % 3
        ax = axes[row, col]

        cal = calendar.Calendar(firstweekday=6)  # começa no domingo
        month_days = cal.monthdayscalendar(year, month)

        data_plot = []
        for week in month_days:
            row_vals = []
            for day in week:
                if day == 0:
                    row_vals.append(np.nan)
                else:
                    date = pd.Timestamp(year=year, month=month, day=day)
                    val = df.loc[date, "Perc"] if date in df.index else np.nan
                    row_vals.append(val)
            data_plot.append(row_vals)

        sns.heatmap(data_plot, cmap=pal, vmin=0, vmax=0.6,
                    cbar=False, linewidths=0.5, linecolor='white',
                    xticklabels=dias_semana, yticklabels=False, ax=ax)

        ax.set_title(meses_pt[month - 1], fontsize=12, loc='left')
        ax.tick_params(bottom=False, left=False)

    # Ajusta layout
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Adiciona barra de cores à direita
    cbar_ax = fig.add_axes([0.96, 0.25, 0.015, 0.5])
    norm = plt.Normalize(vmin=0, vmax=0.6)
    sm = plt.cm.ScalarMappable(cmap=pal, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Percentual de Atrasos")

    plt.show()

# =========================================
# 5. Executar para companhia
# =========================================
for cia in ["AA"]:
    baseCalendario_mes_completo(df_stats, cia)
