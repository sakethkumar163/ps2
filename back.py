from flask import Flask, render_template, request, jsonify
import datacube
import io
import odc.algo
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datacube.utils.cog import write_cog
import base64
import numpy as np

from deafrica_tools.plotting import display_map, rgb
dc = datacube.Datacube(app="04_Plotting")
# 15.85828652, 80.78694696
# 15.75418332, 81.02203692

from geopy.geocoders import Nominatim

def mangrove_analysis(dataset, mvi):
    ndvi = (dataset.B08_10m - dataset.B04_10m) / (dataset.B08_10m + dataset.B04_10m)
    # ndwi=(dataset.B08_10m - dataset.B11_20m) / (dataset.B08_10m + dataset.B11_20m)
    ndvi_threshold = 0.4
    mangrove_mask_ndvi = np.where(ndvi > ndvi_threshold, 1, 0)
    mvi_threshold = 4
    mangrove_mask_mvi = np.where(mvi > mvi_threshold, 1, 0)

    # regular_mask= np.where(ndvi <= 0.7, True, False)

    # closed_mask=np.where(ndvi > 0.7, True, False)

    # healthy_mask=np.where(ndwi>0.2,True,False)

    # unhealthy_mask=np.where(ndwi<0.2,True,False)

    mangrove = np.logical_and(mangrove_mask_ndvi, mangrove_mask_mvi)
    # regular=np.logical_and(mangrove, regular_mask)
    # closed=np.logical_and(mangrove, closed_mask)
    # healthy=np.logical_and(mangrove, healthy_mask)
    # unhealthy=np.logical_and(mangrove, unhealthy_mask)
    pixel_area = abs(dataset.geobox.affine[0] * dataset.geobox.affine[4])

    data = []

    for i in range(mangrove.shape[0]):
        mangrove_cover_area = np.sum(mangrove[i]) * pixel_area
        data.append(mangrove_cover_area/1000000)
        # data_time = str(ndvi.time[i].values).split("T")[0]
        # print(data_time)
        # new_data_time = data_time
        
        # Calculate the total mangrove cover area
        # regular_cover_area=np.sum(regular[i])*pixel_area
        # closed_cover_area=np.sum(closed[i])*pixel_area
        # healthy_cover_area=np.sum(healthy[i])*pixel_area
        # unhealthy_cover_area=np.sum(unhealthy[i])*pixel_area

        # original_array = np.where(ndvi > 0, 1, 0)
        # original = np.sum(original_array[i]) * pixel_area

        # print("Mangrove", mangrove_cover_area/1000000, "sq. km")
        # print("Regular",regular_cover_area/1000000, "sq. km")
        # print("Closed",closed_cover_area/1000000, "sq. km")
        # print("Healthy",healthy_cover_area/1000000, "sq. km")
        # print("Unhealthy",unhealthy_cover_area/1000000, "sq. km")
        # print("Original", original/1000000, "sq. km")
        # print()
    return data

def mang_change(times, query):
    # Load the data for the first time period
    query['time'] = times[0]
    ds1 = dc.load(**query)


    # Compute the MVI for the first time period
    mangrove1 = ((ds1.B08_10m - ds1.B03_10m) / (ds1.B11_20m - ds1.B03_10m+0.5))*(1.5)
    # Set threshold for mangrove detection
    mangrove_thresh = 0.5

    # Create a mangrove mask
    mangrove_mask1 = np.where(mangrove1 > mangrove_thresh, 1, 0)

    # Load the data for the second time period
    query['time'] = times[1]
    ds2 = dc.load(**query)

    # Compute the MVI for the second time period
    mangrove2 = ((ds2.B08_10m - ds2.B03_10m) / (ds2.B11_20m - ds2.B03_10m+0.5))*(1.5)
    # Create a mangrove mask
    mangrove_mask2 = np.where(mangrove2 > mangrove_thresh, 1, 0)

    # Compute the change in mangrove extent
    mangrove_change = mangrove_mask2 - mangrove_mask1

    # Create a colormap
    cmap = plt.get_cmap('PiYG')

    # Plot the change in mangrove extent
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mangrove_change[-1], cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(f'Change in Mangrove Extent from {times[0]} to {times[1]}')
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Change in Mangrove Extent')
    ax.legend(
    [
        Patch(facecolor='lime'),
        Patch(facecolor='fuchsia'),
        Patch(facecolor="palegoldenrod"),
    ],
    ["New mangroves", "Loss of mangroves", "Stable Mangroves"],
    loc="lower right",
)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    # plt.savefig('./static/my_plot.png')
    # Serve the image file in the Flask app
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return img_base64


def get_area_name(latitude, longitude):
    geolocator = Nominatim(user_agent='my-app')  # Initialize the geocoder
    location = geolocator.reverse((latitude, longitude))  # Reverse geocode the coordinates
    if location is not None:
        address_components = location.raw['address']
        city_name = address_components.get('city', '')
        if not city_name:
            city_name = address_components.get('town', '')
        if not city_name:
            city_name = address_components.get('village', '')
        return city_name
    else:
        return "City name not found"

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("main.html")

@app.route('/my_flask_route', methods=['GET', 'POST'])
def my_flask_function():
    if request.method == "POST":
        lmin = request.json['lat_min']
        lmax = request.json['lat_max']
        lnmin = request.json['lng_min']
        lnmax = request.json['lng_max']
        td = request.json['todate']
        fd = request.json['fromdate']
        ind = request.json['index']

        lat_range = (lmin, lmax)
        lon_range = (lnmin, lnmax)
        print(lat_range, lon_range)
        if(td=="" or fd==""):
            query = {
                "product": "s2a_sen2cor_granule",
                "measurements":["B04_10m","B03_10m","B02_10m", "B08_10m", "SCL_20m", "B11_20m", "B8A_20m", "B03_20m"],
                "x":lon_range,
                "y":lat_range,
                "output_crs":'EPSG:6933',
                "resolution":(-30, 30)
            }
        else:
            query = {
                "product": "s2a_sen2cor_granule",
                "measurements":["B04_10m","B03_10m","B02_10m", "B08_10m", "SCL_20m", "B11_20m", "B8A_20m", "B03_20m"],
                "x":lon_range,
                "y":lat_range,
                "time": (fd, td),
                "output_crs":'EPSG:6933',
                "resolution":(-30, 30)
            }
        # display_map(x=lon_range, y=lat_range)
        col = ""
        mi = 0
        ma = 1
        data = []
        try:
            ds = dc.load(**query)
            dataset = ds
            dataset =  odc.algo.to_f32(dataset)
            if(ind == 'NDVI'):
                band_diff = dataset.B08_10m - dataset.B04_10m;
                band_sum = dataset.B08_10m + dataset.B04_10m;
                index = band_diff/band_sum;
                col = "Greens"
            elif(ind == 'NDWI'):
                band_diff = dataset.B03_10m - dataset.B08_10m
                band_sum = dataset.B03_10m + dataset.B08_10m
                index = band_diff / band_sum
                col = "Blues"
            else:
                mi = 1
                ma = 20
                col = "cividis"
                band_diff = dataset.B8A_20m - dataset.B03_10m
                band_sum = dataset.B11_20m - dataset.B03_10m
                index = band_diff / band_sum
                data = mangrove_analysis(dataset, index)
        except Exception as e:
            return jsonify({'error': "No Data Found"})

        # Calculate NDVI and store it as a measurement in the original dataset
        labels = list(map(lambda x: x.split('T')[0], [i for i in np.datetime_as_string(index.time.values).tolist()]))
        # mean_ndvi = selected_times.mean(dim='time')  # Calculate the mean along the 'time' dimension

        # Print the resulting mean_ndvi
        area_name = get_area_name(np.mean(lat_range), np.mean(lon_range))
        # print(area_name)
        # masked_ds_mean = index.mean(dim=['x', 'y'], skipna=True)
        if(ind!="Mangrove Analysis"):
            masked_ds = index.copy()
            masked_ds = masked_ds.where(~np.isinf(masked_ds), drop=False)
            masked_ds_mean = masked_ds.mean(dim=['x', 'y'], skipna=True)
            data = list(map(lambda x:round(x, 4), masked_ds_mean.values.tolist())) 
        plt.figure(figsize=(8, 8))
        subset = index.isel(time=[0, -1])
        subset.plot(col='time', vmin=mi, vmax=ma, col_wrap=2, cmap=col)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        # plt.savefig('./static/my_plot.png')
        # Serve the image file in the Flask app
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        # print({'image': img_base64, 'chman': , 'labels': labels, 'data': data, 'area': area_name})

    # Return the base64 encoded PNG image as JSON
        return jsonify({'image': img_base64, 'chman': mang_change(labels, query), 'labels': labels, 'data': data, 'area': area_name})
    # Calculate the components that make up the NDVI calculation

app.run(host='0.0.0.0', port=5005, debug=True)

