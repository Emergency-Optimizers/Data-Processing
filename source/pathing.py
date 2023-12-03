import utils
import constants

import numpy as np
import pandas as pd
from tqdm import tqdm
import osmnx as ox
import networkx as nx
import os
import geopandas as gpd
import shapely.geometry
import osmnx.distance


class OriginDestination:
    """Class for loading dataset."""
    def __init__(self, dataset_id: str, graph_central_location: tuple[float, float], grap_distance: int, utm_epsg: str):
        """
        Initialize the OD matrix.

        Args:
            dataset_id: Unique identifier for the dataset.
        """
        self.dataset_id = dataset_id
        self.graph_central_location = graph_central_location
        self.grap_distance = grap_distance
        self.utm_epsg = utm_epsg

        df = pd.read_csv(utils.get_enhanced_incidents_path(self.dataset_id), low_memory=False)
        self.file_path = os.path.join(constants.PROJECT_DIRECTORY_PATH, "data", self.dataset_id, "od_matrix.txt")
        
        self.min_row = df["grid_row"].min()
        self.min_col = df["grid_col"].min()
        self.max_row = df["grid_row"].max()
        self.max_col = df["grid_col"].max()
        self.ids = None
        self.matrix: np.ndarray = None

        self.id_to_index: dict = None
        self.num_ids: int = 0

        self.graph = None
        self.node_cache = {}
        self.ids = df["grid_id"].unique()
        self.has_visited = {}

    def build(self):
        if os.path.exists(self.file_path):
            progress_bar = tqdm(desc="Building OD matrix", total=1)
            self.read()
            progress_bar.update(1)
            return

        #self.ids = [utils.row_col_to_id(row, col)
        #            for row in range(self.min_row, self.max_row + 1)
        #            for col in range(self.min_col, self.max_col + 1)]

        self.id_to_index = {id_: index for index, id_ in enumerate(sorted(self.ids))}

        self.num_ids = len(self.ids)
        self.matrix = np.zeros((self.num_ids, self.num_ids), dtype=np.float32)

        self.graph = self.get_graph()

        for origin_id in tqdm(self.ids, desc="Building OD matrix"):
            origin_index = self.id_to_index[origin_id]
            origin_location = utils.id_to_easting_northing(origin_id)

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

                destination_location = utils.id_to_easting_northing(destination_id)

                origin_node = self.get_nearest_node(origin_location[0], origin_location[1])
                destination_node = self.get_nearest_node(destination_location[0], destination_location[1])

                if origin_node == destination_node:
                    self.matrix[origin_index, destination_index] = 0
                    self.matrix[destination_index, origin_index] = 0
                    continue

                shortest_time_path = nx.shortest_path(self.graph, origin_node, destination_node, weight='time')

                total_travel_time = sum(self.graph[u][v][0]['time'] for u, v in zip(shortest_time_path[:-1], shortest_time_path[1:])) * 60

                if total_travel_time != 0:
                    self.matrix[origin_index, destination_index] = total_travel_time
                    self.matrix[destination_index, origin_index] = total_travel_time
                else:
                    self.matrix[origin_index, destination_index] = float("inf")
                    self.matrix[destination_index, origin_index] = float("inf")
        
        self.write()

    def get_nearest_node(self, x, y):
        node_key = (x, y)
        if node_key in self.node_cache:
            return self.node_cache[node_key]

        node = osmnx.distance.nearest_nodes(self.graph, x, y)

        self.node_cache[node_key] = node

        return node
    
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
    
    def get_graph(self):
        graph = ox.graph_from_point(self.graph_central_location, dist=self.grap_distance, network_type="drive", simplify=True, retain_all=False)
        graph = ox.project_graph(graph, to_crs=self.utm_epsg)

        speeds_normal = {
            30: 26.9,
            40: 45.4,
            50: 67.6,
            60: 85.8,
            70: 91.8,
            80: 104.2,
            90: 104.2, # custom
            100: 120.0,
            110: 120.0, # custom
            120: 136.9
        }
        INTERSECTION_PENALTY = 10

        # Adjust the weights of the edges in the graph based on the average speeds and intersection penalty
        for u, v, data in graph.edges(data=True):
            # Use average speeds if available
            if "maxspeed" in data:
                # Take the min of the maxspeed list (in case it's a list)
                if isinstance(data["maxspeed"], list):
                    speed_limits = [speeds_normal.get(int(s), int(s)) for s in data["maxspeed"]]
                    speed_limit = sum(speed_limits) / len(speed_limits)
                else:
                    speed_limit = speeds_normal.get(int(data["maxspeed"]), int(data["maxspeed"]))
                
                # Use average speed if available, else use speed limit
                avg_speed = speed_limit * 0.65
            else:
                avg_speed = 50  # Default speed if not provided
            
            # Calculate time = distance/speed and assign it as the new weight
            # Speeds are in km/h, so converting them to m/min
            data["time"] = data["length"] / (avg_speed * 1000/60)

            # Add intersection penalty if the road segment has an intersection
            if "junction" in graph.nodes[u] or "highway" in graph.nodes[u] and graph.nodes[u]["highway"] == "traffic_signals":
                data["time"] += INTERSECTION_PENALTY / 60
            if "junction" in graph.nodes[v] or "highway" in graph.nodes[v] and graph.nodes[v]["highway"] == "traffic_signals":
                data["time"] += INTERSECTION_PENALTY / 60

        return graph
