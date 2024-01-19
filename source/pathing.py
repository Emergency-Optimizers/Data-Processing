import utils
import constants

import numpy as np
import pandas as pd
from tqdm import tqdm
import osmnx
import networkx
import os
import math
import osmnx.distance
import geopandas as gpd
import shapely.geometry
import gc


class OriginDestination:
    """Class for loading dataset."""
    def __init__(self, dataset_id: str, utm_epsg: str):
        """
        Initialize the OD matrix.

        Args:
            dataset_id: Unique identifier for the dataset.
        """
        self.dataset_id = dataset_id
        self.utm_epsg = utm_epsg
        self.file_path = os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", self.dataset_id, "od_matrix.txt")

        df = pd.read_csv(utils.get_enhanced_incidents_path(self.dataset_id), low_memory=False)
        df2 = pd.read_csv(utils.get_enhanced_depots_path(self.dataset_id), low_memory=False)
        grid_ids = pd.concat([df["grid_id"], df2["grid_id"]])
        self.ids = grid_ids.unique()
        self.ids.sort()
        self.num_ids = len(self.ids)
        self.totalGridsToProcess = int(((self.num_ids - 1) * self.num_ids) / 2)

        self.matrix: np.ndarray = np.zeros((self.num_ids, self.num_ids), dtype=np.float32)

        self.id_to_index = {id_: index for index, id_ in enumerate(self.ids)}

        self.graph: networkx.MultiDiGraph = None
        self.node_cache = {}
        self.has_visited = {}
        self.node_validator = None

    def build(self):
        if os.path.exists(self.file_path):
            progress_bar = tqdm(desc="Building OD matrix", total=1)
            self.read()
            progress_bar.update(1)
            return

        #self.ids = [utils.row_col_to_id(row, col)
        #            for row in range(self.min_row, self.max_row + 1)
        #            for col in range(self.min_col, self.max_col + 1)]

        self.get_graph()
        self.set_graph_weights()

        central_depot_grid_id = 22620006649000
        central_depot_x, central_depot_y = utils.id_to_utm(central_depot_grid_id)
        self.node_validator = osmnx.distance.nearest_nodes(self.graph, central_depot_x, central_depot_y)

        for grid_id in tqdm(self.ids, desc="Caching nodes"):
            self.get_node(grid_id)

        gc.collect()

        progress_bar = tqdm(desc="Building OD matrix", total=self.totalGridsToProcess)

        for origin_id in self.ids:
            origin_index = self.id_to_index[origin_id]
            origin_node = self.get_node(origin_id)

            origin_nodes = []
            destination_nodes = []
            destination_indicies = []

            for destination_id in self.ids:
                if (origin_id, destination_id) in self.has_visited:
                    continue
                else:
                    self.has_visited[(origin_id, destination_id)] = True
                    self.has_visited[(destination_id, origin_id)] = True

                destination_index = self.id_to_index[destination_id]

                if origin_id == destination_id:
                    self.matrix[origin_index, destination_index] = 0
                    continue

                destination_node = self.get_node(destination_id)

                if origin_node == destination_node:
                    distance  = math.dist(utils.id_to_utm(origin_id), utils.id_to_utm(destination_id))
                    speed_km_per_hr = 50
                    distance_km = distance / 1000
                    time_hr = distance_km / speed_km_per_hr
                    travel_time = time_hr * 3600

                    self.matrix[origin_index, destination_index] = travel_time
                    self.matrix[destination_index, origin_index] = travel_time

                    progress_bar.update(1)
                    continue

                origin_nodes.append(origin_node)
                destination_nodes.append(destination_node)
                destination_indicies.append(destination_index)

            shortest_time_paths = osmnx.shortest_path(
                self.graph,
                origin_nodes,
                destination_nodes,
                weight='time',
                cpus=6
            )

            for i, shortest_time_path in enumerate(shortest_time_paths):
                total_travel_time = sum(self.graph[u][v][0]['time'] for u, v in zip(shortest_time_path[:-1], shortest_time_path[1:])) * 60

                self.matrix[origin_index, destination_indicies[i]] = total_travel_time
                self.matrix[destination_indicies[i], origin_index] = total_travel_time

            progress_bar.update(len(shortest_time_paths))
        self.write()

    def write(self):
        with open(self.file_path, "w") as file:
            file.write(','.join(map(str, self.ids)) + '\n')
            for row in self.matrix:
                file.write(','.join(map(str, row)) + '\n')

    def read(self):
        with open(self.file_path, 'r') as file:
            self.ids = list(map(int, file.readline().strip().split(',')))
            matrix = [list(map(float, line.strip().split(','))) for line in file]
        self.matrix = np.array(matrix)

    def get_node(self, grid_id):
        if grid_id in self.node_cache:
            return self.node_cache[grid_id]

        x, y = utils.id_to_utm(grid_id)

        while True:
            node = osmnx.distance.nearest_nodes(self.graph, x, y)
            if networkx.has_path(self.graph, node, self.node_validator) and networkx.has_path(self.graph, self.node_validator, node):
                self.node_cache[grid_id] = node
                return node
            else:
                self.graph.remove_node(node)

    def get_centroid_max_distance(self):
        akershus_gdf: gpd.GeoDataFrame = gpd.read_file(os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_akershus_polygon_epsg4326.geojson"))
        oslo_gdf: gpd.GeoDataFrame = gpd.read_file(os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", "ssb_2019_oslo_polygon_epsg4326.geojson"))

        akershus_gdf['dissolve_field'] = 'Region'
        oslo_gdf['dissolve_field'] = 'Region'
        combined_gdf: gpd.GeoDataFrame = pd.concat([akershus_gdf, oslo_gdf], ignore_index=True)
        gdf: gpd.GeoDataFrame = combined_gdf.dissolve(by='dissolve_field')

        gdf = gdf.to_crs(epsg=32633)
        centroid: shapely.geometry.Point = gdf.geometry.centroid.iloc[0]

        lon, lat = utils.utm_to_geographic(centroid.x, centroid.y)

        max_distance = gdf.geometry.bounds.apply(
            lambda row: max(
                centroid.distance(shapely.geometry.Point(row['minx'], row['miny'])),
                centroid.distance(shapely.geometry.Point(row['minx'], row['maxy'])),
                centroid.distance(shapely.geometry.Point(row['maxx'], row['miny'])),
                centroid.distance(shapely.geometry.Point(row['maxx'], row['maxy']))
            ),
            axis=1
        ).max()

        return (lat, lon), max_distance

    def get_graph(self):
        centroid, distance = self.get_centroid_max_distance()

        self.graph = osmnx.graph_from_point(
            centroid,
            distance,
            dist_type="bbox",
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=False
        )
        self.graph = osmnx.project_graph(self.graph, to_crs=self.utm_epsg)

    def set_graph_weights(self):
        speeds_normal = {
            30: 26.9,
            40: 45.4,
            50: 67.6,
            60: 85.8,
            70: 91.8,
            80: 104.2,
            90: 112.1,
            100: 120.0,
            110: 128.45,
            120: 136.9
        }
        INTERSECTION_PENALTY = 10

        for u, v, data in self.graph.edges(data=True):
            if "maxspeed" in data and data["maxspeed"] != "NO:urban":
                if isinstance(data["maxspeed"], list):
                    speed_limits = [speeds_normal.get(int(s), int(s)) for s in data["maxspeed"]]
                    speed_limit = sum(speed_limits) / len(speed_limits)
                else:
                    speed_limit = speeds_normal.get(int(data["maxspeed"]), int(data["maxspeed"]))

                avg_speed = speed_limit * 0.65
            else:
                avg_speed = 50

            data["time"] = data["length"] / (avg_speed * 1000/60)

            if "junction" in self.graph.nodes[u] or "highway" in self.graph.nodes[u] and self.graph.nodes[u]["highway"] == "traffic_signals":
                data["time"] += INTERSECTION_PENALTY / 60
            if "junction" in self.graph.nodes[v] or "highway" in self.graph.nodes[v] and self.graph.nodes[v]["highway"] == "traffic_signals":
                data["time"] += INTERSECTION_PENALTY / 60
