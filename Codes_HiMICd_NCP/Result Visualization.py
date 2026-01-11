# -*- coding: utf-8 -*-
# %% Fig 3: Scatter plots of predicted and observed near-surface atmospheric moisture indices over the NCP during 2003-2020
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import optimize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

index_name = "SH" #indices including ['AVP', 'DPT', 'MR', 'RH', 'SH', 'VPD']
###------ load sample data ------
Result_data_samPath = r"D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig3 data\sampled_{}.csv".format(index_name)
Result_data_sample = pd.read_csv(Result_data_samPath)
ss = Result_data_sample.filter(regex="Unname")
Result_data_sample = Result_data_sample.drop(ss, axis=1)

x = Result_data_sample["y_test"].dropna()
y = Result_data_sample["LGBM_y_pred"].dropna()

###------ Accuracy Assessment ------
print('R²：', metrics.r2_score(x, y, multioutput='uniform_average'))
print('Mean Absolute Error:', metrics.mean_absolute_error(x, y))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(x, y)))

R_square = metrics.r2_score(x, y, multioutput='uniform_average')
RMSE = np.sqrt(metrics.mean_squared_error(x, y))
MAE = metrics.mean_absolute_error(x, y)
###------ scatter plot ------
x = x.values.ravel()
y = y.values.ravel()

plt.rc('font', family='Arial')
plt.rc('axes', linewidth="1", grid='False')

x2 = np.linspace(-100, 100)
y2 = x2

# fitted line
def f_1(x, A, B):
    return A * x + B

A1, B1 = optimize.curve_fit(f_1, x, y)[0]
y3 = A1 * x + B1

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
# Calculate the point density
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

scatter = ax.scatter(x=x, y=y, marker='.', c=z * 100, edgecolors='none', s=12, label=None, cmap='Spectral_r', zorder=2)

cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30)
cbar.ax.locator_params(nbins=8)
cbar.ax.tick_params(which='both', direction='in', labelsize=40, left=False)
cbar.ax.yaxis.get_offset_text().set_size(36)
ax.plot(x2, y2, color='k', linewidth=1, linestyle='--', alpha=0.7, zorder=1)
ax.plot(x, y3, color='r', linewidth=2, linestyle='-', alpha=0)

textstr = '\n'.join((r'$\mathrm{R}^2$ = %.3f' % R_square, r'$\mathrm{MAE}$ = %.3f' % MAE, r'$\mathrm{RMSE}$ = %.3f' % RMSE))

plt.gca().text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=45, verticalalignment='bottom', horizontalalignment='right', fontname='Arial',
               linespacing=1.5, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.0))

plt.grid(False, alpha=0.01)

if index_name == "RH":
    start_t = -3
    end_t = 102
elif index_name  == "AVP":
    start_t = -3
    end_t = 42
elif index_name  == "VPD":
    start_t = -3
    end_t = 47
elif index_name == "DPT":
    start_t = -45
    end_t = 35
elif index_name == "MR":
    start_t = -2
    end_t = 27
elif index_name == "SH":
    start_t = -2
    end_t = 27
ax.set_xlim((start_t, end_t))
ax.set_ylim((start_t, end_t))

if index_name == "RH":
    tick_values = np.arange(0, 101, 20)
if index_name == "AVP":
    tick_values = np.arange(0, 45, 10)
if index_name == "VPD":
    tick_values = np.arange(0, 50, 10)
if index_name == "DPT":
    tick_values = np.arange(-30, 35, 15)
if index_name == "MR" or index_name == "SH":
    tick_values = np.arange(0, 30, 5)

ax.set_xticks(tick_values)
ax.set_yticks(tick_values)
ax.tick_params(axis="x", labelsize=44, pad=10)
ax.tick_params(axis="y", labelsize=44, pad=10)
if index_name == "RH":
    plt.xlabel('Observed Values (%)',fontsize=45)
    plt.ylabel('Predicted Values (%)',fontsize=45)
if index_name == "AVP" or index_name == "VPD":
    plt.xlabel('Observed Values (hPa)',fontsize=45)
    plt.ylabel('Predicted Values (hPa)',fontsize=45)
if index_name == "DPT":
    plt.xlabel('Observed Values (°C)',fontsize=45)
    plt.ylabel('Predicted Values (°C)',fontsize=45)
if index_name == "MR" or index_name == "SH":
    plt.xlabel('Observed Values (g/kg)',fontsize=45)
    plt.ylabel('Predicted Values (g/kg)',fontsize=45)
# save
plt.tight_layout()
plt.savefig(r'D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig3 data\Results\{}.pdf'.format(index_name))



# %% Fig 4: Annual and monthly R2, MAE and RMSE heatmap of six atmospheric moisture indices over NCP during 2003-2020 (Transposed)
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

###------ load data ------
indicator = "R2"  # including ["R2", "MAE", "RMSE"]
# indicator = "MAE"  # including ["R2", "MAE", "RMSE"]
# indicator = "RMSE"  # including ["R2", "MAE", "RMSE"]
# scale = "monthly" # including ["monthly", "yearly"]
scale = "yearly" # including ["monthly", "yearly"]
yearly_accurary_path = rf"D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig4 data\{indicator}_{scale}_accuracy.csv"
yearly_accurary = pd.read_csv(yearly_accurary_path)

ss = yearly_accurary.filter(regex="Unname")
yearly_accurary = yearly_accurary.drop(ss, axis=1)
yearly_accurary.set_index(["index"], inplace=True)

# 转置数据
yearly_accurary = yearly_accurary.T

original_values = yearly_accurary.copy()

colors = ['lightyellow', 'salmon', 'maroon']
cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1', colors)

if indicator == "R2":
    vmin, vmax = 0.86, 1.0
else:
    log_data = np.log10(yearly_accurary.replace(0, np.nan))
    vmin, vmax = log_data.min().min(), log_data.max().max()

def smart_format_log(x):
    if np.isnan(x):
        return ""
    return f"{x:.2f}"

###------ figure ------
if scale == "yearly":
    plt.figure(figsize=(10, 8))
else:  # monthly
    plt.figure(figsize=(12, 8))

if indicator == "R2":
    ax = sns.heatmap(
        yearly_accurary,
        annot=np.round(yearly_accurary, 3),
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap1,
        center=None,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'shrink': 1, 'aspect': 25, 'pad': 0.02, 'label': "log10(Value)" if indicator != "R2" else "Value"},
        annot_kws={"size": 14}
    )
else:
    log_data = np.log10(yearly_accurary.replace(0, np.nan))
    ax = sns.heatmap(
        log_data,
        annot=np.vectorize(smart_format_log)(log_data),
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap1,
        center=None,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={'shrink': 1, 'aspect': 25, 'pad': 0.02, 'label': 'log10(Value)'},
        annot_kws={"size": 14}
    )

texts = ax.texts
values = original_values.values.flatten()

colorbar = ax.collections[0].colorbar
colorbar.ax.tick_params(labelsize=16)
if indicator == "R2":
    label_text = r"$R^2$"
elif indicator == "MAE":
    label_text = "MAE (log10)"
else:
    label_text = "RMSE (log10)"

colorbar.set_label(label_text, fontsize=16)

if indicator == "R2":
    tick_positions = np.linspace(vmin, vmax, 5)
    tick_labels = [f'{x:.2f}' for x in tick_positions]
    colorbar.set_ticks(tick_positions)
    colorbar.set_ticklabels(tick_labels)

# 设置坐标轴标签（转置后交换了x和y轴的含义）
ax.set_xlabel('Atmospheric Moisture Index', fontsize=18)
if scale == "yearly":
    ax.set_ylabel('Year', fontsize=18)
elif scale == "monthly":
    ax.set_ylabel('Month', fontsize=18)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=0, ha='center')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)

x_labels = yearly_accurary.columns.tolist()
if len(x_labels) > 10:
    selected_x_positions = range(0, len(x_labels), max(1, len(x_labels)//8))
    ax.set_xticks([pos + 0.5 for pos in selected_x_positions])
    ax.set_xticklabels([x_labels[pos] for pos in selected_x_positions],
                      fontsize=12, rotation=45, ha='right')

y_labels = yearly_accurary.index.tolist()
if len(y_labels) > 15:
    selected_y_positions = range(0, len(y_labels), max(1, len(y_labels)//10))
    ax.set_yticks([pos + 0.5 for pos in selected_y_positions])
    ax.set_yticklabels([y_labels[pos] for pos in selected_y_positions],
                      fontsize=12, rotation=0)

plt.tight_layout()
plt.savefig(rf'D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig4 data\Results\{indicator}_{scale}_transposed.pdf', dpi=300)

plt.show()


# %% Fig 5: Spatial distribution of R2, MAE and RMSE values for six atmospheric moisture
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42


index_name = "AVP"
# index_name = "RH"
# index_name = "VPD"
# index_name = "DPT"
# index_name = "MR"
# index_name = "SH"

indicator = "RMSE"  # including ["R2", "MAE", "RMSE"]

Index_By_Id_path = fr"D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig5 data\{index_name}_metrics_by_station.csv"
Index_By_Id = pd.read_csv(Index_By_Id_path)
Index_By_Id = Index_By_Id.loc[:, ~Index_By_Id.columns.str.contains('^Unnamed')]

pointshp = gpd.read_file(r'D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig5 data\point_shp.shp')
MapShp = gpd.read_file(r'D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig5 data\area.shp')

pointshp["R2"] = np.nan
pointshp["MAE"] = np.nan
pointshp["RMSE"] = np.nan

for i, Aid in enumerate(pointshp["id"]):
    if Aid in Index_By_Id["id"].values:
        AStation = Index_By_Id[Index_By_Id["id"] == Aid]
        pointshp.at[i, "R2"] = AStation["R2"].values[0]
        pointshp.at[i, "MAE"] = AStation["MAE"].values[0]
        pointshp.at[i, "RMSE"] = AStation["RMSE"].values[0]

plt.rc('axes', unicode_minus=False)
plt.rc('axes', linewidth=2.0, grid=False)
plt.rc('font', family='Microsoft YaHei', size=15)

fig, ax = plt.subplots(figsize=(15, 12))
ax.set_alpha(0.8)

if indicator == "R2":
    colors = ['#fff5f5',  'salmon', 'maroon']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1', colors)
    data_values = pointshp[indicator].dropna()
    vmin = max(0.6, data_values.min() * 0.95)
    vmax = min(1.0, data_values.max() * 1.05)
elif indicator == "MAE":
    colors = ['maroon', 'salmon', '#fff5f5']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1', colors)
    data_values = pointshp[indicator].dropna()
    vmin = data_values.min() * 0.95
    vmax = data_values.max() * 1.05
elif indicator == "RMSE":
    colors = ['maroon', 'salmon', '#fff5f5']
    cmap1 = mcolors.LinearSegmentedColormap.from_list('cmap1', colors)
    data_values = pointshp[indicator].dropna()
    vmin = data_values.min() * 0.95
    vmax = data_values.max() * 1.05

if indicator == "R2":
    vmin = max(vmin, 0)
    vmax = min(vmax, 1)
else:
    vmin = max(vmin, 0)

norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

MapShp.geometry.plot(ax=ax, facecolor='#FFF9F1', edgecolor='dimgrey', alpha=1, label='Mainland China')

pointdf_ = gpd.GeoDataFrame(pointshp, geometry=gpd.points_from_xy(pointshp["lon"], pointshp["lat"]))
value = pointdf_[indicator].values
x = pointdf_["lon"].values
y = pointdf_["lat"].values

scatter = ax.scatter(x, y, c=value, cmap=cmap1, norm=norm, edgecolors='gray', linewidths=1, s=170, alpha=1, label='Meteorological station')

cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.020, aspect=40)

# Format the tick labels
from matplotlib.ticker import MaxNLocator
cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
if indicator == "R2":
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
else:
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Configure tick appearance
cbar.ax.tick_params(which='both',
                    direction='in',
                    labelsize=45,
                    left=False)
cbar.ax.yaxis.get_offset_text().set_size(36)

if index_name == 'DPT':
    label_dict = {"R2": " ", "MAE": "(°C)", "RMSE": "(°C)"}
if index_name == 'RH':
    label_dict = {"R2": " ", "MAE": "(%)", "RMSE": "(%)"}
if index_name == 'AVP' or index_name == 'VPD':
    label_dict = {"R2": " ", "MAE": "(hPa)", "RMSE": "(hPa)"}
if index_name == 'MR' or index_name == 'SH':
    label_dict = {"R2": " ", "MAE": "(g/kg)", "RMSE": "(g/kg)"}

if index_name == 'RH':
    cbar.ax.text(2.7, 1.08, label_dict.get(indicator, indicator), fontsize=45, ha='center', transform=cbar.ax.transAxes)

if index_name == 'AVP' or index_name == 'VPD':
    cbar.ax.text(3.2, 1.04, label_dict.get(indicator, indicator), fontsize=45, ha='center', transform=cbar.ax.transAxes)

if index_name == 'DPT':
    cbar.ax.text(3, 1.04, label_dict.get(indicator, indicator), fontsize=45, ha='center', transform=cbar.ax.transAxes)

if index_name == 'MR' or index_name == 'SH':
    cbar.ax.text(3.5, 1.06, label_dict.get(indicator, indicator), fontsize=45, ha='center', transform=cbar.ax.transAxes)

ax.set_ylim(34, 41)
ax.set_yticks(np.arange(36, 42, 2))
ax.set_xlim(113, 121)
ax.set_xticks(np.arange(114, 122, 2))

if index_name in ['RH', 'AVP', 'VPD']:
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
elif index_name in ['DPT', 'MR', 'SH']:
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f °E'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f °N'))

if index_name in ['MR', 'AVP', 'VPD', 'SH']:
    ax.tick_params(axis='y', labelsize=45, rotation=90, pad=5, left=False, labelleft=False)
if index_name in ['RH', 'DPT']:
    ax.tick_params(axis='y', labelsize=45, rotation=90, pad=5)

ax.tick_params(axis='x', labelsize=45, pad=10, length=15, width=2)
ax.tick_params(axis='y', labelsize=45, pad=10, length=15, width=2)

# scalebar
def add_line_scale_with_labels(ax, length_km, loc=(0.85, 0.1), linewidth=0.8, text_size=20, color='black'):
    length_deg = length_km * 0.00898

    x_center = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * loc[0] -0.8
    y_bottom = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * loc[1]

    x_left = x_center - length_deg / 2
    x_mid = x_center
    x_right = x_center + length_deg / 2

    tick_height = 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 20

    ax.plot([x_left-0.3, x_right+0.3], [y_bottom, y_bottom], color=color, linewidth=linewidth, solid_capstyle='butt')

    for x_pos, label in zip([x_left-0.3, x_mid, x_right+0.3], [0, length_km // 2, length_km]):
        label_str = f'{int(label)} km' if label == length_km else f'{int(label)}'
        ax.plot([x_pos , x_pos], [y_bottom - tick_height+0.06, y_bottom + tick_height+0.06], color=color, linewidth=linewidth)

        ax.text(x_pos, y_bottom + 2 * tick_height+0.055, label_str, ha='center', va='bottom', fontsize=text_size, color=color)

if index_name == "RH":
    add_line_scale_with_labels(ax, length_km=300, loc=(0.75, 0.06), linewidth=3, text_size=45, color='black')

plt.tight_layout()
save_path = fr'D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig5 data\Results\{index_name}_{indicator}.pdf'
plt.savefig(save_path, dpi=300)
plt.show()


# %% Fig 6: Spatial patterns of six moisture indices over NCP from 2003 to 2020
import numpy as np
import xarray as xr
import geopandas as gpd
from osgeo import gdal
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors


index_name = "RH"
# index_name = "AVP"
# index_name = "VPD"
# index_name = "DPT"
# index_name = "MR"
# index_name = "SH"

date ='2008-1-3'
# date ='2013-8-13'

###------ load data ------
MapShp = gpd.read_file(r'D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig6 data\area.shp')
path = r"D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig6 data\{}_{}_LGBM.tif".format(index_name, date)
tifdata01 = gdal.Open(path)
bands = tifdata01.RasterCount
if bands < 1:
    print("without bands！")
else:
    band = tifdata01.GetRasterBand(1)
    A_band = band.ReadAsArray()
del path, tifdata01, bands, band

A_band[A_band < -9000] = np.nan
A_band[A_band > 9000] = np.nan

data_values = A_band[~np.isnan(A_band)]  # 移除NaN值
vmin = data_values.min()
vmax = data_values.max()
vmin = 1
vmax = 4
###------ plot ------
# 设置绘图样式
plt.rc('axes', unicode_minus=False)
plt.rc('axes', linewidth=2.0, grid=False)
fig, ax = plt.subplots(figsize=(15, 12))
ax.set_alpha(0.8)

ax.set_ylim(34, 41)
ax.set_yticks(np.arange(36, 42, 2))
ax.set_xlim(113, 121)
ax.set_xticks(np.arange(114, 122, 2))

if index_name == 'RH' or index_name == 'AVP' or index_name == 'VPD':
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
if index_name == 'DPT' or index_name == 'MR' or index_name == 'SH':
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f °E'))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f °N'))
if index_name == 'MR' or index_name == 'AVP' or index_name == 'VPD' or index_name == 'SH':
    ax.tick_params(axis='y', labelsize=45, rotation=90, pad=5, left=False, labelleft=False)
if index_name == 'RH' or index_name == 'DPT':
    ax.tick_params(axis='y', labelsize=45, rotation=90, pad=5)
ax.tick_params(axis='x', labelsize=45, pad=5)

cmap_slope = "BrBG"
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

extent = (113, 121.0, 34, 41.01)
im1 = ax.imshow(A_band, extent=extent, norm=norm, cmap=cmap_slope)

MapShp.geometry.plot(ax=ax, facecolor='none', edgecolor='dimgrey', alpha=1, linewidth=1)

if index_name == 'RH' or index_name == 'AVP' or index_name == 'VPD' :
    cax = fig.add_axes([0.85, 0.025, 0.03, 0.9])
else:
    cax = fig.add_axes([0.85, 0.075, 0.03, 0.9])


cbar = fig.colorbar(im1, cax=cax, orientation='vertical', extend='both', shrink=1)


cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
cbar.ax.tick_params(which='both', direction='in', labelsize=45)

label_dict = {"AVP": "(hPa)","VPD": "(hPa)","DPT":"(°C)","RH":"(%)","MR": "(g/kg)","SH": "(g/kg)"}
if index_name == 'RH':
    cbar.ax.text(1.87, 1.02, label_dict.get(index_name, index_name), fontsize=45, ha='center', va='bottom', transform=cbar.ax.transAxes)

if index_name == 'DPT':
    cbar.ax.text(2.2, 1.02, label_dict.get(index_name, index_name), fontsize=45, ha='center', va='bottom', transform=cbar.ax.transAxes)

if index_name == 'AVP' or index_name == 'VPD' :
    cbar.ax.text(2.5, 1.02, label_dict.get(index_name, index_name), fontsize=45, ha='center', va='bottom', transform=cbar.ax.transAxes)

if index_name == 'MR' or index_name == 'SH':
    cbar.ax.text(2.2, 1.04, label_dict.get(index_name, index_name), fontsize=45, ha='center', va='bottom', transform=cbar.ax.transAxes)

def add_line_scale_with_labels(ax, length_km=300, loc=(0.75, 0.06), linewidth=3, text_size=45, color='black'):
    length_deg = length_km * 0.00898

    x_center = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * loc[0]-0.8
    y_bottom = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * loc[1]

    x_left = x_center - length_deg / 2
    x_right = x_center + length_deg / 2

    tick_height = 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 20

    ax.plot([x_left-0.3, x_right+0.3], [y_bottom, y_bottom], color=color, linewidth=linewidth)

    for x_pos, label in zip([x_left-0.3, x_center, x_right+0.3], [0, length_km // 2, length_km]):
        label_str = f'{int(label)} Km' if label == length_km else f'{int(label)}'
        ax.plot([x_pos, x_pos], [y_bottom - tick_height + 0.06, y_bottom + tick_height + 0.06], color=color, linewidth=linewidth)

        ax.text(x_pos, y_bottom + 2 * tick_height + 0.055, label_str, ha='center', va='bottom', fontsize=text_size, color=color)

if index_name == "RH":
    add_line_scale_with_labels(ax)

export_path = r'D:\pycharm_code\HiMICd-NCP-main\Data Samples_HiMICd_NCP\Fig6 data\Results\{}_{}.pdf'.format(index_name, date)
plt.tight_layout()
plt.savefig(export_path, dpi=300, bbox_inches='tight')
plt.show()



# %%[Fig 7: Comparison with existing datasets]
import numpy as np
import xarray as xr
import geopandas as gpd
from osgeo import gdal
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd

index_name = "RH"
date = '2018-9-20'

###------ Load data ------
MapShp = gpd.read_file(r'D:\pycharm_code\HiTIC-NCP-main\Data Samples_HiMICd_NCP\Fig7 data\area.shp')
path = r"D:\pycharm_code\HiTIC-NCP-main\Data Samples_HiMICd_NCP\Fig7 data\{}\{}_LGBM.tif".format(index_name, date)

tifdata01 = gdal.Open(path)
bands = tifdata01.RasterCount
if bands < 1:
    print("No bands found!")
else:
    band = tifdata01.GetRasterBand(1)
    A_band = band.ReadAsArray()
del path, tifdata01, bands, band

# Data cleaning
A_band[A_band < -9000] = np.nan
A_band[A_band > 9000] = np.nan

###------ Load station data ------
# Read station latitude and longitude information
points_df = pd.read_csv(r'D:\pycharm_code\HiTIC-NCP-main\Data Samples_HiMICd_NCP\Fig7 data\point.csv')
# Read station observation data
obs_df = pd.read_csv(r'D:\pycharm_code\HiTIC-NCP-main\Data Samples_HiMICd_NCP\Fig7 data\combined_data_2018.csv')

# Merge two tables by id
merged_df = pd.merge(points_df, obs_df, on='id', how='inner')
station_data = merged_df

# Extract longitude, latitude, and RH values
station_lons = station_data['lon'].values
station_lats = station_data['lat'].values
station_values = station_data['RH'].values

###------ Set zoom area ------
# Define the area to zoom in
zoom_extent = [115.7, 117.1, 39.6, 40.6]
full_extent = (113, 121.0, 34, 41.01)

###------ Extract zoom area data and calculate range ------
# Calculate indices of zoom area in the array
original_shape = A_band.shape
lon_res = (full_extent[1] - full_extent[0]) / original_shape[1]
lat_res = (full_extent[3] - full_extent[2]) / original_shape[0]

x_start = int((zoom_extent[0] - full_extent[0]) / lon_res)
x_end = int((zoom_extent[1] - full_extent[0]) / lon_res)
y_start = int((full_extent[3] - zoom_extent[3]) / lat_res)
y_end = int((full_extent[3] - zoom_extent[2]) / lat_res)

# Ensure indices are within valid range
x_start = max(0, x_start)
x_end = min(original_shape[1], x_end)
y_start = max(0, y_start)
y_end = min(original_shape[0], y_end)

# Extract zoom area data
zoom_data = A_band[y_start:y_end, x_start:x_end]

###------ Calculate color bar range based on zoom area data ------
data_values = zoom_data[~np.isnan(zoom_data)]  # Remove NaN values
vmin = data_values.min()
vmax = data_values.max()
# Uncomment below if fixed range is needed
vmin = 35
vmax = 95

print(f"Zoom area data range: {vmin:.2f} - {vmax:.2f}")

###------ Plot ------
# Set plot style
plt.rc('axes', unicode_minus=False)
plt.rc('axes', linewidth=2.0, grid=False)
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_alpha(0.8)

# Set display range to zoom area only
ax.set_xlim(zoom_extent[0], zoom_extent[1])
ax.set_ylim(zoom_extent[2], zoom_extent[3])
ax.set_aspect('equal')

# Hide axes
ax.set_axis_off()

cmap_slope = "BrBG"
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Create image using original data but color mapping based on zoom area range
im1 = ax.imshow(A_band,
                extent=full_extent,
                norm=norm,  # Use norm calculated based on zoom area
                cmap=cmap_slope)

# Plot administrative boundaries in zoom area
MapShp.geometry.plot(ax=ax,
                     facecolor='none',
                     edgecolor='dimgrey',
                     alpha=1,
                     linewidth=1.5)

# Filter stations within zoom area
in_zoom_mask = ((station_lons >= zoom_extent[0]) & (station_lons <= zoom_extent[1]) &
                (station_lats >= zoom_extent[2]) & (station_lats <= zoom_extent[3]))

zoom_station_lons = station_lons[in_zoom_mask]
zoom_station_lats = station_lats[in_zoom_mask]
zoom_station_values = station_values[in_zoom_mask]

# Plot station scatter plot, using RH values as colors
scatter = ax.scatter(zoom_station_lons, zoom_station_lats,
                     c=zoom_station_values,
                     cmap=cmap_slope,
                     norm=norm,
                     s=300,           # Point size
                     edgecolors='black',  # Point border color
                     linewidth=2,   # Border width
                     zorder=10)       # Ensure stations are displayed on top layer

# Save image
export_path = r'D:\pycharm_code\HiTIC-NCP-main\Data Samples_HiMICd_NCP\Fig7 data\Results\{}_{}_HiMIC_zoomed.pdf'.format(index_name, date)

plt.tight_layout()
plt.savefig(export_path, dpi=300, bbox_inches='tight')
plt.show()


