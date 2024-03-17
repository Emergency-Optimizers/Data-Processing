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
import itertools


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

        self.boundry_north = df["latitude"].max()
        self.boundry_south = df["latitude"].min()
        self.boundry_east = df["longitude"].max()
        self.boundry_west = df["longitude"].min()

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

        self.get_graph()

        params = {
            "default_intersection_penalty": 2,
            "traffic_signal_penalty": 5,
            "road_type_factors": {
                "motorway": 0.9704026402720763,
                "trunk": 0.9872728848094453,
                "primary": 0.9434615680765118,
                "secondary": 0.49062545330175955,
                "tertiary": 0.9327873614166583,
                "unclassified": 0.8502498894729571,
                "residential": 0.7543590170147566,
                "living_street": 0.8078248863334735,
            },
        }
        params["road_type_factors"] = self.gen_linkers(params["road_type_factors"], link_factor=0.2630284973443693)
        self.set_graph_weights_v2(**params)

        central_depot_grid_id = 22620006649000
        central_depot_x, central_depot_y = utils.id_to_utm(central_depot_grid_id)
        self.node_validator = osmnx.distance.nearest_nodes(self.graph, central_depot_x, central_depot_y)

        for grid_id in tqdm(self.ids, desc="Caching nodes"):
            self.get_node(grid_id)

        gc.collect()

        od_pairs = [(origin_id, destination_id) 
                    for origin_id, destination_id in itertools.product(self.ids, repeat=2)
                    if origin_id != destination_id]

        batch_size = math.ceil(len(od_pairs) / 10)
        od_batches = [od_pairs[i:i + batch_size] for i in range(0, len(od_pairs), batch_size)]

        progress_bar = tqdm(desc="Building OD matrix", total=len(od_pairs))

        for batch in od_batches:
            origin_nodes = [self.get_node(od_pair[0]) for od_pair in batch]
            destination_nodes = [self.get_node(od_pair[1]) for od_pair in batch]

            shortest_paths = osmnx.shortest_path(
                self.graph,
                origin_nodes,
                destination_nodes,
                weight='time',
                cpus=5
            )

            for i, path in enumerate(shortest_paths):
                if path is None:
                    continue

                origin_id, destination_id = batch[i]
                origin_index = self.id_to_index[origin_id]
                destination_index = self.id_to_index[destination_id]

                total_travel_time = sum(self.graph[u][v][0]['time'] for u, v in zip(path[:-1], path[1:])) * 60
                self.matrix[origin_index, destination_index] = total_travel_time
                self.matrix[destination_index, origin_index] = total_travel_time

            progress_bar.update(len(shortest_paths))

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
            node = osmnx.distance.nearest_nodes(self.graph, x + 500, y + 500)
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
        """centroid, distance = self.get_centroid_max_distance()

        self.graph = osmnx.graph_from_point(
            centroid,
            distance,
            dist_type="bbox",
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=False
        )"""

        self.graph = osmnx.graph_from_bbox(
            north=self.boundry_north,
            south=self.boundry_south,
            east=self.boundry_east,
            west=self.boundry_west,
            network_type="drive",
            simplify=True,
            retain_all=False,
            truncate_by_edge=False
        )

        self.graph = osmnx.project_graph(self.graph, to_crs=self.utm_epsg)

    def set_graph_weights(self, intersection_penalty=10, secret_scary_factor=0.65, use_ambulance_speeds=True):

        if use_ambulance_speeds:
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
        else:
            speeds_normal = {
                30: 30,
                40: 40,
                50: 50,
                60: 60,
                70: 70,
                80: 80,
                90: 90,
                100: 100,
                110: 110,
                120: 120
            }

        for u, v, data in self.graph.edges(data=True):
            if "maxspeed" in data and data["maxspeed"] != "NO:urban":
                if isinstance(data["maxspeed"], list):
                    speed_limits = [speeds_normal.get(int(s), int(s)) for s in data["maxspeed"]]
                    speed_limit = sum(speed_limits) / len(speed_limits)
                else:
                    speed_limit = speeds_normal.get(int(data["maxspeed"]), int(data["maxspeed"]))

                avg_speed = speed_limit * secret_scary_factor
            else:
                avg_speed = 50

            data["time"] = data["length"] / (avg_speed * 1000/60)

            if "junction" in self.graph.nodes[u] or "highway" in self.graph.nodes[u] and self.graph.nodes[u]["highway"] == "traffic_signals":
                data["time"] += intersection_penalty / 60
            if "junction" in self.graph.nodes[v] or "highway" in self.graph.nodes[v] and self.graph.nodes[v]["highway"] == "traffic_signals":
                data["time"] += intersection_penalty / 60

    def set_graph_weights_v2(
        self,
        default_intersection_penalty=10,
        traffic_signal_penalty=15,
        road_type_factors=None,
        use_ambulance_speeds=False
    ):
        if road_type_factors is None:
            road_type_factors = {
                "residential": 0.75,
                "tertiary": 0.8,
                "secondary": 0.85,
                "primary": 0.9,
                "motorway": 0.95
            }

        if use_ambulance_speeds:
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
        else:
            speeds_normal = {
                30: 30,
                40: 40,
                50: 50,
                60: 60,
                70: 70,
                80: 80,
                90: 90,
                100: 100,
                110: 110,
                120: 120
            }

        for u, v, data in self.graph.edges(data=True):
            road_type = data.get("highway", "unknown")
            factor = self.get_adjustment_factor(road_type, road_type_factors, 1.0)

            if "maxspeed" in data and data["maxspeed"] != "NO:urban":
                if isinstance(data["maxspeed"], list):
                    speed_limits = [speeds_normal.get(int(s), int(s)) for s in data["maxspeed"]]
                    speed_limit = sum(speed_limits) / len(speed_limits)
                else:
                    speed_limit = speeds_normal.get(int(data["maxspeed"]), int(data["maxspeed"]))

                # Apply road type specific factor if available, else use a default factor
                avg_speed = speed_limit
            else:
                # Apply default speed where specific speed limits are not available
                avg_speed = 50
            
            avg_speed *= factor

            data["time"] = data["length"] / (avg_speed * 1000 / 60)

            intersection_penalty_u = traffic_signal_penalty if "highway" in self.graph.nodes[u] and self.graph.nodes[u]["highway"] == "traffic_signals" else default_intersection_penalty
            intersection_penalty_v = traffic_signal_penalty if "highway" in self.graph.nodes[v] and self.graph.nodes[v]["highway"] == "traffic_signals" else default_intersection_penalty

            # Adjusting time for intersections
            if "junction" in self.graph.nodes[u] or "highway" in self.graph.nodes[u]:
                data["time"] += intersection_penalty_u / 60  # Convert penalty to minutes
            if "junction" in self.graph.nodes[v] or "highway" in self.graph.nodes[v]:
                data["time"] += intersection_penalty_v / 60  # Convert penalty to minutes

    def get_adjustment_factor(self, road_type, road_type_factors, default_factor):
        if isinstance(road_type, list):
            factors = []

            for rt in road_type:
                if rt in road_type_factors:
                    factors.append(road_type_factors.get(rt, default_factor))

            if factors:
                return sum(factors) / len(factors)
            return default_factor
        else:
            # Return the factor for the single road type or the default
            return road_type_factors.get(road_type, default_factor)

    def gen_linkers(self, road_type_factors, link_factor):
        linkables = [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
        ]

        for linkable in linkables:
            if linkable in road_type_factors:
                road_type_factors[linkable + "_link"] = road_type_factors[linkable] * link_factor
        
        return road_type_factors

